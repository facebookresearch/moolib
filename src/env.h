/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "logging.h"
#include "pythonserialization.h"
#include "shm.h"
#include "tensor.h"
#include "util.h"

#include <atomic>
#include <list>
#include <new>

namespace moolib {

struct SharedMapEntry {
  SharedArray<char> key;
  SharedArray<int64_t> shape;
  SharedArray<std::byte> data;
  size_t elements;
  size_t itemsize;
  char dtype;
};

struct BatchData {
  SharedArray<SharedMapEntry> data;
};

constexpr size_t maxClients = 0x100;
constexpr size_t maxEnvs = 0x1000;
constexpr size_t maxBuffers = 4;

template<typename T, size_t Capacity>
struct SharedQueue {
  std::array<T, Capacity> buffer;
  std::atomic_size_t top = 0;
  std::atomic_size_t bot = 0;
  bool empty() {
    return top == bot;
  }
  size_t size() {
    return top - bot;
  }
  size_t capacity() {
    return Capacity;
  }
  template<typename A>
  void push(A&& a) {
    buffer[top++ % Capacity] = std::forward<A>(a);
  }
  T pop() {
    return buffer[bot++ % Capacity];
  }
};

struct Shared {
  size_t size;
  std::atomic_size_t allocated = sizeof(*this);
  void init(size_t size) {
    new (this) Shared();
    this->size = size;
  }

  struct Buffer {

    struct ClientInput {
      std::atomic_size_t nStepsIn = 0;
      std::atomic_size_t resultOffset = 0;
    };
    struct alignas(64) ClientOutput {
      std::atomic_size_t nStepsOut = 0;
    };

    struct EnvInput {
      std::atomic<uint32_t> action = 0;
    };

    size_t size = 0;
    size_t stride = 0;

    std::array<ClientInput, maxClients> clientInputs;
    std::array<EnvInput, maxEnvs> envInputs{};
    std::array<ClientOutput, maxClients> clientOutputs;

    std::atomic_bool batchAllocated;
    std::atomic_bool batchAllocating;
    BatchData batchData;
  };

  alignas(64) std::atomic_size_t clients = 0;
  std::array<Buffer, maxBuffers> buffers;

  struct ClientInput {
    SharedSemaphore semaphore;
    SharedQueue<int, maxBuffers> queue;
  };

  struct alignas(64) ClientOutput {
    SharedSemaphore semaphore;
  };

  std::array<ClientInput, maxClients> clientIn;
  std::array<ClientOutput, maxClients> clientOut;

  std::byte* allocateNonAligned(size_t n) {
    size_t offset = allocated.fetch_add(n, std::memory_order_relaxed);
    if (offset + n > size) {
      throw std::runtime_error("Out of space in shared memory buffer");
    }
    return (std::byte*)this + offset;
  }

  template<typename T>
  SharedArray<T> allocate(size_t n) {
    size_t offset = allocated.fetch_add(sizeof(T) * n, std::memory_order_relaxed);
    if (offset + n > size) {
      throw std::runtime_error("Out of space in shared memory buffer");
    }
    return {n, {offset}};
  }

  template<typename T>
  SharedArray<T> allocateAligned(size_t n) {
    size_t offset = allocated.load(std::memory_order_relaxed);
    size_t newOffset;
    do {
      newOffset = (offset + 63) / 64 * 64;
    } while (!allocated.compare_exchange_weak(offset, newOffset + sizeof(T) * n, std::memory_order_relaxed));
    if (newOffset + n > size) {
      throw std::runtime_error("Out of space in shared memory buffer");
    }
    return {n, {newOffset}};
  }

  template<typename T>
  SharedArray<T> allocateString(std::basic_string_view<T> str) {
    static_assert(std::is_trivially_copyable_v<T>);
    auto r = allocate<T>(str.size());
    std::memcpy(r(this), str.data(), sizeof(T) * str.size());
    return r;
  }
};

struct BatchBuilderHelper {
  std::unordered_map<std::string_view, bool> added;
  struct I {
    std::string key;
    std::vector<int64_t> shape;
    size_t elements;
    size_t itemsize;
    char dtype;
  };
  std::vector<I> fields;
  void add(std::string_view key, size_t dims, const ssize_t* shape, size_t itemsize, char dtype) {
    if (std::exchange(added[key], true)) {
      throw std::runtime_error("key " + std::string(key) + " already exists in batch!");
    }
    size_t elements = 1;
    for (size_t i = 0; i != dims; ++i) {
      elements *= shape[i];
    }
    fields.emplace_back();
    auto& f = fields.back();
    f.key = key;
    f.shape.assign(shape, shape + dims);
    f.elements = elements;
    f.itemsize = itemsize;
    f.dtype = dtype;
  }
};

struct Env {
  py::object env_;
  py::object reset_;
  py::object step_;

  uint64_t steps = 0;

  std::atomic<bool> terminate_ = false;

  std::vector<std::vector<uint32_t>> prevActions_;

  Env(py::object env) {
    py::gil_scoped_acquire gil;
    env_ = std::move(env);
    reset_ = env_.attr("reset");
    step_ = env_.attr("step");
  }

  ~Env() {
    py::gil_scoped_acquire gil;
    env_ = {};
    reset_ = {};
    step_ = {};
  }

  void allocateBatch(Shared* shared, BatchData& batch, const py::dict& obs) {

    BatchBuilderHelper bb;

    for (auto& [key, value] : obs) {
      auto [str, stro] = rpc::pyStrView(key);
      py::array arr = py::reinterpret_borrow<py::object>(value);
      bb.add(str, arr.ndim(), arr.shape(), arr.itemsize(), arr.dtype().kind());
    }

    bb.add("done", 0, nullptr, 1, 'b');
    bb.add("reward", 0, nullptr, 4, 'f');

    batch.data = shared->allocate<SharedMapEntry>(bb.fields.size());
    for (size_t i = 0; i != bb.fields.size(); ++i) {
      auto& f = bb.fields[i];
      auto& d = batch.data(shared, i);
      d.key = shared->allocateString(std::string_view(f.key));
    }
    for (size_t i = 0; i != bb.fields.size(); ++i) {
      auto& f = bb.fields[i];
      auto& d = batch.data(shared, i);
      d.data = shared->allocateAligned<std::byte>(f.itemsize * f.elements * maxEnvs);
      d.shape = shared->allocate<int64_t>(f.shape.size());
      for (size_t i2 = 0; i2 != f.shape.size(); ++i2) {
        d.shape(shared, i2) = f.shape[i2];
      }
      d.elements = f.elements;
      d.itemsize = f.itemsize;
      d.dtype = f.dtype;
    }
    log.debug("allocated %d bytes\n", (int)shared->allocated);
  }

  void fillBatch(Shared* shared, BatchData& batch, size_t batchIndex, std::string_view key, void* src, size_t len) {
    auto* map = batch.data(shared);
    for (size_t i = 0; i != batch.data.size; ++i) {
      auto& v = map[i];
      if (v.key.view(shared) == key) {
        std::byte* dst = v.data(shared);
        dst += v.itemsize * v.elements * batchIndex;
        if (len != v.itemsize * v.elements) {
          throw std::runtime_error("fill batch size mismatch");
        }
        std::memcpy(dst, src, v.itemsize * v.elements);
        return;
      }
    }
    throw std::runtime_error(std::string(key) + ": batch key not found");
  }

  void step(Shared* shared, size_t bufferIndex, size_t batchIndex) {
    ++steps;
    if (prevActions_.size() <= bufferIndex) {
      prevActions_.resize(bufferIndex + 1);
    }
    auto& prevActions = prevActions_[bufferIndex];
    if (prevActions.size() <= batchIndex) {
      prevActions.resize(batchIndex + 1);
    }
    uint32_t prevAction = prevActions[batchIndex];
    uint32_t action = prevAction;
    auto& sa = shared->buffers[bufferIndex].envInputs[batchIndex].action;
    uint32_t timeCheckCounter = 0x100000;
    std::optional<std::chrono::steady_clock::time_point> waitTime;
    do {
      action = sa.load();
      if (terminate_) {
        return;
      }
      if (--timeCheckCounter == 0) {
        timeCheckCounter = 0x100000;
        if (!waitTime) {
          waitTime = std::chrono::steady_clock::now();
        } else if (std::chrono::steady_clock::now() - *waitTime >= std::chrono::seconds(120)) {
          throw std::runtime_error("Timed out waiting for env action");
        }
      }
    } while (action == prevAction);
    prevActions[batchIndex] = action;
    action -= prevAction + 1;
    {
      py::gil_scoped_acquire gil;
      bool done;
      float reward;
      py::dict obs;
      py::object rawObs;
      if (steps == 1) {
        done = false;
        reward = 0.0f;
        rawObs = reset_();
      } else {
        py::tuple tup = step_(action);
        rawObs = tup[0];
        reward = (py::float_)tup[1];
        done = (py::bool_)tup[2];
        if (done) {
          // log.debug("episode done after %d steps with %g total reward\n", episodeStep, episodeReturn);
          rawObs = reset_();
        }
      }
      if (py::isinstance<py::dict>(rawObs)) {
        obs = rawObs.cast<py::dict>();
      } else {
        obs["state"] = rawObs.cast<py::array>();
      }
      auto& buffer = shared->buffers[bufferIndex];
      auto& batch = buffer.batchData;
      if (!buffer.batchAllocated.load()) {
        if (buffer.batchAllocating.exchange(true)) {
          while (!buffer.batchAllocated)
            ;
        } else {
          allocateBatch(shared, batch, obs);
          buffer.batchAllocated = true;
        }
      }
      fillBatch(shared, batch, batchIndex, "done", &done, sizeof(bool));
      fillBatch(shared, batch, batchIndex, "reward", &reward, sizeof(float));
      for (auto& [key, value] : obs) {
        auto [str, stro] = rpc::pyStrView(key);
        py::array arr(py::reinterpret_borrow<py::object>(value));
        fillBatch(shared, batch, batchIndex, str, (float*)arr.data(), arr.nbytes());
      }
    }
  }
};

struct EnvBatch {
  py::object envInit_;
  std::list<Env> envs;
  EnvBatch() = default;
  EnvBatch(py::object envInit) : envInit_(std::move(envInit)) {}
  ~EnvBatch() {
    py::gil_scoped_acquire gil;
    envInit_ = {};
  }
  void step(size_t size, Shared* shared, size_t bufferIndex, size_t batchIndex) {
    while (envs.size() < size) {
      py::gil_scoped_acquire gil;
      envs.emplace_back(envInit_());
    }
    for (auto& v : envs) {
      v.step(shared, bufferIndex, batchIndex);
      ++batchIndex;
    }
  }
};

struct EnvRunner {

  std::atomic_bool terminate_ = false;

  std::thread runThread;

  std::array<EnvBatch, 10> batch;
  bool running_ = false;

  EnvRunner(py::object envInit) {
    for (auto& b : batch) {
      b.envInit_ = envInit;
    }
  }

  ~EnvRunner() {
    py::gil_scoped_release gil;
    terminate_ = true;
    for (auto& v : batch) {
      for (auto& v2 : v.envs) {
        v2.terminate_ = true;
      }
    }
    if (runThread.joinable()) {
      runThread.join();
    }
  }

  bool running() {
    return running_;
  }

  struct SetNotRunning {
    EnvRunner* me;
    ~SetNotRunning() {
      me->running_ = false;
    }
  };

  void start(std::string serverAddress) {
    running_ = true;
    runThread = std::thread([this, serverAddress]() { run(serverAddress); });
  }

  void run(std::string serverAddress) {
    SetNotRunning t{this};
    running_ = true;

    SharedMemory shm(serverAddress);
    Shared& shared = shm.as<Shared>();

    int myIndex = shared.clients++;

    log.debug("my index is %d\n", myIndex);

    if ((size_t)myIndex > maxClients) {
      throw std::runtime_error("Client index is too high");
    }
    auto lastUpdate = std::chrono::steady_clock::now();

    while (true) {
      auto& in = shared.clientIn[myIndex];
      if (!in.queue.empty()) {
        size_t bufferIndex = in.queue.pop();
        auto& buffer = shared.buffers[bufferIndex];
        auto* input = &buffer.clientInputs[myIndex];
        auto* output = &buffer.clientOutputs[myIndex];
        size_t stepsDone = output->nStepsOut.load();
        size_t nSteps = input->nStepsIn.load();
        if (nSteps != stepsDone) {
          lastUpdate = std::chrono::steady_clock::now();
          size_t offset = input->resultOffset;
          batch.at(bufferIndex).step(nSteps - stepsDone, &shared, bufferIndex, offset);
          output->nStepsOut.store(nSteps);
          shared.clientOut[myIndex].semaphore.post();
          continue;
        }
      } else {
        in.semaphore.wait_for(std::chrono::seconds(1));
      }
      if (terminate_.load(std::memory_order_relaxed)) {
        break;
      }
      auto now = std::chrono::steady_clock::now();
      if (now - lastUpdate >= std::chrono::seconds(1800)) {
        log.debug("EnvRunner timed out\n");
        terminate_ = true;
        break;
      }
    }
  }
};

struct EnvStepperFuture {
  struct EnvStepper* stepper = nullptr;
  int bufferIndex;
  size_t size;
  size_t stride;
  Shared::Buffer* buffer;
  py::object result() const;
};

struct EnvStepper {
  std::mutex mutex;
  std::atomic_bool terminate_ = false;
  int numClients_ = 0;
  std::optional<SharedMemory> shm;
  Shared* shared = nullptr;
  std::array<std::atomic_bool, maxBuffers> bufferBusy{};
  std::array<std::atomic_bool, maxBuffers> bufferStarted{};
  std::array<std::map<std::string, rpc::Tensor>, maxBuffers> outputMap;
  std::array<rpc::Tensor, maxBuffers> actionPinned{};
  EnvStepper(std::string shmName, int numClients);
  EnvStepperFuture step(int bufferIndex, py::object action);
};

struct EnvPool {

  struct Impl;
  std::unique_ptr<Impl> impl;

  EnvPool(py::object envInit, int numProcesses);
  ~EnvPool();

  int spawn(std::string shmName);

  std::unique_ptr<EnvStepper> spawn();
};

} // namespace moolib
