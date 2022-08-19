
#pragma once

#include "rpc.h"

#include "fmt/printf.h"

#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

namespace rpc {

struct iovec {
  void* iov_base = nullptr;
  size_t iov_len = 0;
};

struct CachedReader;

struct SocketImpl;
struct Socket {
  std::shared_ptr<SocketImpl> impl;
  Socket();
  Socket(const Socket&) = delete;
  Socket(Socket&& n);
  ~Socket();
  Socket& operator=(const Socket&) = delete;
  Socket& operator=(Socket&& n);
  static Socket Unix();
  static Socket Tcp();

  void close();

  void listen(std::string_view address);
  void accept(Function<void(Error*, Socket)> callback);
  void connect(std::string_view address, Function<void(Error*)> callback);

  void writev(const iovec* vec, size_t veclen, Function<void(Error*)> callback);

  void setOnRead(Function<void(Error*, std::unique_lock<SpinMutex>*)> callback);

  size_t readv(const iovec* vec, size_t veclen);

  void sendFd(int fd, Function<void(Error*)> callback);
  int recvFd(CachedReader& reader);

  std::string localAddress() const;
  std::string remoteAddress() const;

  int nativeFd() const;
};

struct CachedReader {
  std::vector<iovec> iovecs;
  size_t iovecsOffset;
  Socket* socket;
  size_t bufferFilled = 0;
  size_t bufferOffset = 0;
  std::vector<char> buffer;
  CachedReader(Socket& socket) : socket(&socket) {}
  void newRead() {
    iovecs.clear();
    iovecsOffset = 0;
  }
  void addIovec(const iovec& v) {
    iovecs.push_back(v);
  }
  void addIovec(void* dst, size_t len) {
    iovecs.push_back(iovec{dst, len});
  }
  void startRead() {
    if (buffer.size() < 4096) {
      buffer.resize(4096);
    }
    iovecsOffset = 0;
    size_t skip = bufferFilled - bufferOffset;
    if (skip) {
      //fmt::printf("skip is %d\n", skip);
      size_t left = skip;
      size_t offset = bufferOffset;
      char* src = buffer.data() + bufferOffset;
      const char* end = buffer.data() + bufferFilled;
      for (auto& v : iovecs) {
        size_t n = std::min(left, v.iov_len);
        std::memcpy(v.iov_base, src, n);
        //fmt::printf("filled %d/%d bytes from buffer\n", n, v.iov_len);
        v.iov_base = (char*)v.iov_base + n;
        v.iov_len -= n;
        if (v.iov_len == 0) {
          ++iovecsOffset;
        }
        src += n;
        assert(src <= end);
        left -= n;
        if (left == 0) {
          break;
        }
      }
      bufferOffset = src - buffer.data();
      // fmt::printf("buffer %d/%d\n", bufferOffset, bufferFilled);
      // fmt::printf("iovecsOffset %d/%d\n", iovecsOffset, iovecs.size());
      if (bufferOffset == bufferFilled) {
        bufferOffset = 0;
        bufferFilled = 0;
      } else assert(iovecsOffset == iovecs.size());
    }
    iovecs.push_back({buffer.data() + bufferFilled, buffer.size() - bufferFilled});
  }
  bool done() {
    assert(iovecsOffset < iovecs.size());
    if (iovecsOffset && iovecsOffset == iovecs.size() - 1) {
      return true;
    }
    // std::string s;
    // for (auto& v : iovecs) {
    //   if (!s.empty()) {
    //     s += ", ";
    //   }
    //   s += fmt::sprintf("%d bytes", v.iov_len);
    // }
    // fmt::printf("done() called with %d iovecs (offset %d): %s\n", iovecs.size(), iovecsOffset, s);
    size_t n = socket->readv(iovecs.data() + iovecsOffset, iovecs.size() - iovecsOffset);
    //fmt::printf("socket readv returned %d\n", n);
    bool retval = false;
    size_t i = iovecsOffset;
    size_t e = iovecs.size() - 1;
    for (; i != e; ++i) {
      auto& v = iovecs[i];
      if (n >= v.iov_len) {
        //fmt::printf("buffer %d completely filled\n", i);
        n -= v.iov_len;
        v.iov_len = 0;
        ++iovecsOffset;
        if (n == 0) {
          ++i;
          break;
        }
      } else {
        //fmt::printf("buffer %d filled %d/%d\n", n, v.iov_len);
        v.iov_base = (char*)v.iov_base + n;
        v.iov_len -= n;
        return false;
      }
    }
    if (i == e) {
      assert(iovecsOffset == iovecs.size() - 1);
      bufferFilled += n;
      //fmt::printf("%d bytes leftover for buffer, buffer is now %d/%d\n", n, bufferOffset, bufferFilled);
      if (bufferFilled == buffer.size()) {
        buffer.resize(buffer.size() * 2);
        fmt::printf("buffer full, resized to %d\n", buffer.size());
      }
      return true;
    }
    assert(iovecsOffset <= iovecs.size() - 1);
    return false;
  }
  void* readBufferPointer(size_t len) {
    if (bufferFilled - bufferOffset >= len) {
      size_t o = bufferOffset;
      bufferOffset += len;
      return buffer.data() + o;
    }
    if (buffer.size() < bufferOffset + len) {
      buffer.resize(bufferOffset + len);
    }
    newRead();
    startRead();
    done();
    if (bufferFilled - bufferOffset >= len) {
      size_t o = bufferOffset;
      bufferOffset += len;
      return buffer.data() + o;
    }
    return nullptr;
  }
  bool readCopy(void* dst, size_t len) {
    void* ptr = readBufferPointer(len);
    if (ptr) {
      std::memcpy(dst, ptr, len);
      return true;
    } else {
      return false;
    }
  }
};

} // namespace rpc
