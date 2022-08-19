
#include "ipc.h"

#include <limits.h>
#include <sys/uio.h>
#include <unistd.h>

namespace rpc {

extern thread_local std::array<const char*, 4000> names;
extern thread_local std::array<uint64_t, 4000> times;

extern thread_local bool callbackScheduledFromBackend;

namespace ipc {

static size_t computeStorageNbytes(IntArrayRef sizes, IntArrayRef strides, size_t itemsize_bytes) {
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  size_t size = 1;
  for (size_t i = 0; i < sizes.size(); i++) {
    if (sizes[i] == 0) {
      return 0;
    }
    size += strides[i] * (sizes[i] - 1);
  }
  return size * itemsize_bytes;
}

void Listener::accept(Function<void(Error*, std::shared_ptr<Connection>)> callback) {
  socket.accept([this, callback = std::move(callback)](Error* error, Socket socket) {
    if (error) {
      callback(error, nullptr);
    } else {
      auto connection = std::make_shared<Connection>(std::move(socket));
      callback(nullptr, std::move(connection));
    }
  });
}

std::string Listener::localAddress() const {
  return socket.localAddress();
}

const uint32_t sigSocketData = 0x39eb69f4;

SharedBufferHandle toShared(SharedBufferHandle h) {
  return h;
}
SharedBufferHandle toShared(BufferHandle h) {
  return SharedBufferHandle(h.release());
}

template<typename Buffer>
void Connection::write(Buffer buffer, Function<void(Error*)> callback) {
  uint32_t size = buffer->size;
  if ((size_t)size != buffer->size) {
    throw Error("write: buffer is too large (size does not fit in 32 bits)");
  }
  thread_local std::vector<iovec> buffers;
  buffers.resize(2 + buffer->nTensors);
  buffers[1].iov_base = buffer->data();
  buffers[1].iov_len = (std::byte*)(buffer->tensorMetaDataOffsets() + buffer->nTensors) - buffer->data();
  auto* tensors = buffer->tensors();
  for (size_t i = 0; i != buffer->nTensors; ++i) {
    Tensor& tensor = tensors[i].tensor;
    size_t length = computeStorageNbytes(tensor.sizes(), tensor.strides(), tensor.itemsize());
    if (tensor.is_cuda()) {
      throw Error("IPC can not send CUDA tensors");
    } else {
      buffers[2 + i].iov_base = tensor.data_ptr();
      buffers[2 + i].iov_len = length;
    }
  }
  auto buffer0 = serializeToBuffer(SerializeFunction([&](auto& v) {
    v(sigSocketData, (uint32_t)buffers.size() - 1, size);
    for (size_t i = 1; i != buffers.size(); ++i) {
      v(buffers[i].iov_len);
    }
  }));

  buffers[0].iov_base = buffer0->data();
  buffers[0].iov_len = buffer0->size;

  socket.writev(
      buffers.data(), buffers.size(),
      [buffer0 = std::move(buffer0), buffer = std::move(buffer), callback = std::move(callback)](Error* error) {
        if (callback) {
          callback(error);
        }
      });
}

template void Connection::write(BufferHandle, Function<void(Error*)>);
template void Connection::write(SharedBufferHandle, Function<void(Error*)>);

void Connection::close() {
  socket.close();
  readCallback = nullptr;
}

void Connection::read(Function<void(Error*, BufferHandle)> callback) {
  readCallback = std::move(callback);
  socket.setOnRead([this](Error* error, auto* lock) mutable {
    if (error) {
      readCallback(error, nullptr);
      return;
    }

    int count = 0;

    static constexpr int stateZero = 0;
    static constexpr int stateSocketReadSizes = 1;
    static constexpr int stateSocketReadIovecs = 2;
    static constexpr int stateAllDone = 3;
    
    // struct Foo {
    //   int* c;
    //   Foo(int* c) : c(c) {}
    //   ~Foo() {
    //     fmt::printf("~Foo %d\n", *c);
    //   }
    // } foo(&count);

    while (true) {
      //fmt::printf("pid %d ipc fd %d enter state %d\n", ::getpid(), socket.nativeFd(), readState);
      switch (readState) {
      default:
        close();
        fmt::printf("ipc fd %d exit after %d - due to close\n", socket.nativeFd(), count);
        return;
      case stateZero: {
        if (!socket.read(tmpReadBuffer.data(), 12)) {
          //fmt::printf("ipc fd %d exit after %d - stateZero\n", socket.nativeFd(), count);
          return;
        }
        uint32_t numBuffers, bufferSize;
        uint32_t recvSignature;
        deserializeBuffer(tmpReadBuffer.data(), 12, recvSignature, numBuffers, bufferSize);
        switch (recvSignature) {
        case sigSocketData:
          break;
        default:
          readState = -1;
          Error e("bad signature");
          readCallback(&e, nullptr);
          fmt::printf("exit after %d - bad sig\n", count);
          std::abort();
          return;
        }
        // if (numBuffers == 0 || numBuffers > 20) {
        //   fmt::printf("exit after %d - bad numbuffers %#x\n", count, numBuffers);
        //   std::abort();
        // }
        // if (bufferSize > 1000) {
        //   fmt::printf("exit after %d - bad bufferSize %#x\n", count, bufferSize);
        //   std::abort();
        // }
        buffer = makeBuffer(bufferSize, numBuffers - 1);
        bufferSizes.clear();
        bufferSizes.resize(numBuffers);
        readState = stateSocketReadSizes;
        [[fallthrough]];
      }
      case stateSocketReadSizes: {
        if (!socket.read(bufferSizes.data(), bufferSizes.size() * sizeof(size_t))) {
          //fmt::printf("ipc fd %d exit after %d - stateSocketReadSizes\n", socket.nativeFd(), count);
          return;
        }
        if (bufferSizes[0] !=
            size_t((std::byte*)(buffer->tensorMetaDataOffsets() + buffer->nTensors) - buffer->data())) {
          readState = -1;
          Error e("bad buffer size");
          readCallback(&e, nullptr);
          fmt::printf("ipc fd %d exit after %d - bad sizes\n", socket.nativeFd(), count);
          std::abort();
          return;
        }
        allocators.clear();
        iovecs.clear();
        for (size_t i = 0; i != bufferSizes.size(); ++i) {
          if (i != 0) {
            allocators.emplace_back(rpc::kCPU, bufferSizes[i]);
          }
          iovec v;
          v.iov_len = bufferSizes[i];
          v.iov_base = i == 0 ? buffer->data() : allocators.back().data();
          iovecs.push_back(v);
        }
        readState = stateSocketReadIovecs;
        [[fallthrough]];
      }
      case stateSocketReadIovecs:
        if (!socket.readv(iovecs.data(), iovecs.size())) {
          //fmt::printf("ipc fd %d exit after %d - stateSocketReadIovecs\n", socket.nativeFd(), count);
          return;
        } else {
          readState = stateAllDone;
          [[fallthrough]];
        }
      case stateAllDone: {
        auto* offsets = buffer->tensorMetaDataOffsets();
        auto* tensors = buffer->tensors();
        auto* data = buffer->data();
        size_t nTensors = buffer->nTensors;
        size_t len = buffer->size;
        for (size_t i = 0; i != nTensors; ++i) {
          Tensor& t = tensors[i].tensor;
          decltype(t.scalar_type()) dtype;
          decltype(t.sizes()) sizes;
          decltype(t.strides()) strides;
          deserializeBufferPart(data + offsets[i], len > offsets[i] ? len - offsets[i] : 0, dtype, sizes, strides);
          t = allocators[i].set(dtype, sizes, strides);
        }
        readState = stateZero;

        // static std::chrono::steady_clock::time_point last;
        // auto now = std::chrono::steady_clock::now();
        // float t = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(now - last).count();
        // last = now;
        // fmt::printf("scheduled! time since last - %g\n", t * 1000);

        // rpc::callbackScheduledFromBackend = true;
        readCallback(nullptr, std::move(buffer));
        // scheduler.run([connection = shared_from_this(), buf = std::move(buffer)]() mutable {
        //   //fmt::printf("callback handled in %d\n", ::gettid());
        //   //std::memset(&times, 0, sizeof(times));
        //   //auto start = std::chrono::steady_clock::now();
        //   callbackScheduledFromBackend = true;
        //   connection->readCallback(nullptr, std::move(buf));
        //   callbackScheduledFromBackend = false;
        //   //auto now = std::chrono::steady_clock::now();
        //   //float t = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(now - start).count();
        //   //fmt::printf("callback handled in %g\n", t * 1000);

        //   // if (t * 1000 > 1) {
        //   //   uint64_t maxvalue = 0;
        //   //   for (size_t i = 0; i != times.size(); ++i) {
        //   //     if (times[i] > maxvalue) {
        //   //       maxvalue = times[i];
        //   //     }
        //   //   }
        //   //   std::string str;
        //   //   for (size_t i = 0; i != times.size(); ++i) {
        //   //     if (times[i] > 0) {
        //   //       str += fmt::sprintf("%s: %d (%g%%)\n", names[i], times[i], times[i] / (double)maxvalue * 100);
        //   //     }
        //   //   }
        //   //   fmt::printf("%s", str);
        //   // }
        // });
        ++count;
        break;
      }
      }
    }
    //fmt::printf("exit after %d - function exit\n", count);
  });
}

std::string Connection::localAddress() const {
  return socket.localAddress();
}

std::string Connection::remoteAddress() const {
  return socket.remoteAddress();
}

Connection::~Connection() {}

std::shared_ptr<Listener> UnixContext::listen(std::string_view addr) {
  auto listener = std::make_shared<Listener>(Socket::Unix());
  listener->socket.listen(addr);
  return listener;
}

std::shared_ptr<Connection> UnixContext::connect(std::string_view addr) {
  auto connection = std::make_shared<Connection>(Socket::Unix());

  connection->socket.connect(addr, [connection](Error* e) {
    if (e) {
      // connection->close();
    }
  });

  return connection;
}

std::shared_ptr<Listener> TcpContext::listen(std::string_view addr) {
  auto listener = std::make_shared<Listener>(Socket::Tcp());
  listener->socket.listen(addr);
  return listener;
}

std::shared_ptr<Connection> TcpContext::connect(std::string_view addr) {
  auto connection = std::make_shared<Connection>(Socket::Tcp());

  connection->socket.connect(addr, [connection](Error* e) {
    if (e) {
      // connection->close();
    }
  });

  return connection;
}

} // namespace ipc
} // namespace rpc
