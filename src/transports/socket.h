
#pragma once

#include "rpc.h"

#include <memory>
#include <mutex>
#include <string_view>

namespace rpc {

struct iovec {
  void* iov_base = nullptr;
  size_t iov_len = 0;
};

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

  bool read(void* dst, size_t size);
  bool readv(const iovec* vec, size_t veclen);

  void sendFd(int fd, Function<void(Error*)> callback);
  int recvFd();

  std::string localAddress() const;
  std::string remoteAddress() const;

  int nativeFd() const;
};

} // namespace rpc
