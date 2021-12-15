/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "env.h"
#include "logging.h"

#include <csignal>
#include <sys/types.h>
#include <unistd.h>

#include "tensor.h"

namespace moolib {

std::atomic<int> liveThreads = 0;
struct ThreadCounter {
  ThreadCounter() {
    ++liveThreads;
  }
  ~ThreadCounter() {
    --liveThreads;
  }
};
thread_local ThreadCounter threadCounter;

struct Pipe {
  int readfd = -1;
  int writefd = -1;
  Pipe() {
    int fds[2];
    if (::pipe(fds)) {
      int err = errno;
      perror("pipe");
      fatal("pipe failed with error %d", err);
    }
    readfd = fds[0];
    writefd = fds[1];
  }
  ~Pipe() {
    if (readfd != -1) {
      ::close(readfd);
    }
    if (writefd != -1) {
      ::close(writefd);
    }
  }
  Pipe(const Pipe&) = delete;
  Pipe(Pipe&&) = delete;
  Pipe& operator=(const Pipe&) = delete;
  Pipe& operator=(Pipe&&) = delete;
  void closeWrite() {
    ::close(writefd);
    writefd = -1;
  }
  void closeRead() {
    ::close(readfd);
    readfd = -1;
  }

  enum class Result {
    success,
    eof,
    timeout,
  };

  Result read(void* buf, size_t len, std::chrono::steady_clock::duration timeout) {
    std::byte* ptr = (std::byte*)buf;
    fd_set readfds;
    auto abstimeout = std::chrono::steady_clock::now() + timeout;
    timeval tv;
    for (; len; timeout = abstimeout - std::chrono::steady_clock::now()) {
      auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(timeout);
      auto seconds = std::chrono::duration_cast<std::chrono::seconds>(timeout);
      tv.tv_sec = seconds.count();
      tv.tv_usec = (microseconds - seconds).count();
      FD_ZERO(&readfds);
      FD_SET(readfd, &readfds);
      int r = ::select(readfd + 1, &readfds, nullptr, &readfds, &tv);
      if (r < 0) {
        int err = errno;
        if (err == EINTR) {
          continue;
        }
        perror("select");
        fatal("select failed with error %d", err);
      } else if (r == 0) {
        return Result::timeout;
      }
      r = ::read(readfd, ptr, len);
      if (r == 0) {
        return Result::eof;
      } else if (r < 0) {
        int err = errno;
        if (err == EINTR) {
          continue;
        }
        perror("read");
        fatal("read failed with error %d", err);
      }
      len -= r;
      ptr += r;
    }
    return Result::success;
  }
  bool read(void* buf, size_t len) {
    std::byte* ptr = (std::byte*)buf;
    while (len) {
      int r = ::read(readfd, ptr, len);
      if (r == 0) {
        return false;
      } else if (r < 0) {
        int err = errno;
        if (err == EINTR) {
          continue;
        }
        perror("read");
        fatal("read failed with error %d", err);
      }
      len -= r;
      ptr += r;
    }
    return true;
  }
  bool write(const void* buf, size_t len) {
    const std::byte* ptr = (const std::byte*)buf;
    while (len) {
      int r = ::write(writefd, ptr, len);
      if (r == 0) {
        return false;
      } else if (r < 0) {
        int err = errno;
        if (err == EINTR) {
          continue;
        }
        perror("write");
        fatal("write failed with error %d", err);
      }
      len -= r;
      ptr += r;
    }
    return true;
  }
};

template<typename F>
void fork(F&& f) {
  (void)threadCounter;
  if (liveThreads != 1) {
    fatal(
        "Refusing to fork because there are %d threads!\n"
        "Please instantiate environments as early as possible, "
        "before creating any threads or trainers.",
        (int)liveThreads);
  }
  auto pid = ::fork();
  if (pid < 0) {
    int err = errno;
    perror("fork");
    fatal("fork failed with error %d", err);
  }
  if (pid == 0) {
    std::forward<F>(f)();
    std::exit(0);
  }
}

struct EnvPool::Impl {
  Pipe requestPipe;
  Pipe responsePipe;
};

EnvPool::EnvPool(py::object envInit, int numProcesses) {
  impl = std::make_unique<Impl>();
  fork([&]() {
    signal(SIGINT, SIG_IGN);
    impl->requestPipe.closeWrite();
    impl->responsePipe.closeRead();
    size_t len;
    std::string address;
    Pipe keepalivePipe;
    while (impl->requestPipe.read(&len, sizeof(len))) {
      address.resize(len);
      if (!impl->requestPipe.read(address.data(), address.size())) {
        break;
      }
      for (int i = 0; i != numProcesses; ++i) {
        fork([&]() {
          keepalivePipe.closeWrite();
          EnvRunner env(envInit);
          std::thread terminateThread([&]() {
            int buf;
            while (keepalivePipe.read(&buf, sizeof(buf), std::chrono::seconds(1)) == Pipe::Result::timeout) {
              if (env.terminate_) {
                break;
              }
            }
            env.terminate_ = true;
          });
          Dtor cleanup = [&]() {
            env.terminate_ = true;
            terminateThread.join();

            Py_Exit(0);
            // TODO: signal to main process that we're exiting
          };
          try {
            env.run(address);
          } catch (const pybind11::error_already_set& e) {
            fatal("Error in env: %s\n", e.what());
          }
        });
      }
      int num = numProcesses;
      impl->responsePipe.write(&num, sizeof(num));
    }
  });
  impl->requestPipe.closeRead();
  impl->responsePipe.closeWrite();
}

EnvPool::~EnvPool() {}

int EnvPool::spawn(std::string shmName) {
  size_t len = shmName.size();
  if (!impl->requestPipe.write(&len, sizeof(len))) {
    fatal("Failed to communicate with EnvPool server (1)");
  }
  if (!impl->requestPipe.write(shmName.data(), len)) {
    fatal("Failed to communicate with EnvPool server (2)");
  }
  int num;
  if (!impl->responsePipe.read(&num, sizeof(num))) {
    fatal("Failed to communicate with EnvPool server (3)");
  }
  if (num == 0) {
    fatal("EnvPool server failed to start any environment runners");
  }
  return num;
}

std::unique_ptr<EnvStepper> EnvPool::spawn() {
  std::string name = randomName();
  int n = spawn(name);
  return std::make_unique<EnvStepper>(name, n);
}

EnvStepper::EnvStepper(std::string shmName, int numClients) {
  numClients_ = numClients;
  shm.emplace(shmName);
  shared = &shm->as<Shared>();

  while ((int)shared->clients != numClients) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (terminate_) {
      shm->unlink();
      return;
    }
  }

  shm->unlink();
}

namespace {

async::SchedulerFifo async(1);

} // namespace

EnvStepperFuture EnvStepper::step(int bufferIndex, py::object actionObject) {
  auto actionOpt = rpc::tryFromPython(actionObject);
  if (!actionOpt) {
    throw std::runtime_error(
        "EnvStepper::step function was passed an action argument that could not be converted to a Tensor");
  }
  auto& action = *actionOpt;

  if (action.itemsize() != sizeof(long) ||
      (action.scalar_type() != rpc::Tensor::kInt32 && action.scalar_type() != rpc::Tensor::kInt64)) {
    throw std::runtime_error("EnvStepper::step expected action tensor with data type long");
  }
  if (action.dim() != 1) {
    throw std::runtime_error("EnvStepper::step expected a 1-dimensional tensor");
  }

  if (bufferIndex < 0 || (size_t)bufferIndex >= bufferBusy.size()) {
    throw std::runtime_error(fmt::sprintf("EnvStepper: buffer index (%d) out of range", bufferIndex));
  }

  if (bufferBusy[bufferIndex].exchange(true, std::memory_order_relaxed)) {
    throw std::runtime_error(
        fmt::sprintf("EnvStepper: attempt to step buffer index %d twice concurrently", bufferIndex));
  }

  auto& buffer = shared->buffers[bufferIndex];

  std::optional<rpc::CUDAStream> stream;
  if (action.is_cuda()) {
    stream.emplace(rpc::getCurrentCUDAStream());
  }

  size_t size = action.size(0);
  size_t strideDivisor = numClients_;
  size_t stride = (size + strideDivisor - 1) / strideDivisor;

  async.run([this, action = std::move(action), bufferIndex, size, stride, stream = std::move(stream)]() mutable {
    if (stream) {
      rpc::CUDAStreamGuard sg(*stream);
      auto& pinned = actionPinned[bufferIndex];
      if (!pinned.defined()) {
        pinned = action.cpu().pin_memory();
      }
      pinned.copy_(action, true);
      action = pinned;
      stream->synchronize();
    }

    auto& buffer = shared->buffers[bufferIndex];

    size_t clientIndex = 0;
    for (size_t i = 0; i < size; i += stride, ++clientIndex) {
      int nSteps = std::min(size - i, stride);
      auto& input = buffer.clientInputs[clientIndex];
      input.resultOffset.store(i);
      input.nStepsIn.fetch_add(nSteps);

      auto& in = shared->clientIn[clientIndex];
      if (in.queue.size() >= in.queue.capacity()) {
        fatal("EnvStepper: shared queue is full");
      }
      in.queue.push(bufferIndex);
      in.semaphore.post();
    }

    bufferStarted[bufferIndex] = true;

    auto acc = action.data<long>();

    for (size_t i = 0; i != size; ++i) {
      auto& action = buffer.envInputs[i].action;
      action.store(action.load(std::memory_order_relaxed) + 1 + acc[i], std::memory_order_relaxed);
    }
  });

  return {this, bufferIndex, size, stride, &buffer};
}

py::object EnvStepperFuture::result() const {
  std::optional<py::gil_scoped_release> gil;
  gil.emplace();

  bool terminate_ = false; // ?

  auto& buffer = *this->buffer;

  auto start = std::chrono::steady_clock::now();
  while (!stepper->bufferStarted[bufferIndex].load(std::memory_order_relaxed)) {
    auto now = std::chrono::steady_clock::now();
    if (now - start >= std::chrono::seconds(60)) {
      throw std::runtime_error("Timed out waiting for bufferStarted");
    }
    std::this_thread::yield();
  }
  size_t clientIndex = 0;
  for (size_t i = 0; i < size; i += stride, ++clientIndex) {
    auto& input = buffer.clientInputs[clientIndex];
    auto& output = buffer.clientOutputs[clientIndex];
    size_t prevSteps = input.nStepsIn.load();
    while (true) {
      if (terminate_) {
        return {};
      }
      if (output.nStepsOut.load() == prevSteps) {
        break;
      }
      auto now = std::chrono::steady_clock::now();
      if (now - start >= std::chrono::seconds(1800)) {
        throw std::runtime_error("Timed out waiting for env");
      }
      stepper->shared->clientOut[clientIndex].semaphore.wait_for(std::chrono::seconds(2));
    }
  }

  auto& outputMap = stepper->outputMap[bufferIndex];

  if (outputMap.empty()) {
    auto* shared = stepper->shared;
    auto* src = buffer.batchData.data(shared);
    for (size_t i = 0; i != buffer.batchData.data.size; ++i) {
      auto key = src[i].key.view(shared);
      auto& v = src[i];
      auto dtype = getTensorDType(v.dtype, v.itemsize);
      std::vector<int64_t> sizes(v.shape(shared), v.shape(shared) + v.shape.size);
      sizes.insert(sizes.begin(), (int64_t)size);
      auto t = rpc::from_blob(dtype, {sizes.data(), sizes.size()}, v.data(shared));
      outputMap[std::string(key)] = t;
    }
  }
  stepper->bufferStarted[bufferIndex].store(false, std::memory_order_relaxed);
  stepper->bufferBusy[bufferIndex].store(false, std::memory_order_relaxed);

  auto& m = outputMap;
  gil.reset();
  py::dict r;
  for (auto& [key, value] : m) {
    r[py::str(key)] = rpc::toPython(value);
  }
  return (py::object)std::move(r);
}

} // namespace moolib
