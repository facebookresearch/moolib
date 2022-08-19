
#pragma once

#include "rpc.h"
#include "socket.h"
#include "synchronization.h"

#include "fmt/printf.h"

#include <memory>

namespace rpc {

namespace ipc {

struct Connection;
struct Listener;

struct UnixContext {
  std::shared_ptr<Listener> listen(std::string_view addr);
  std::shared_ptr<Connection> connect(std::string_view addr);
};

struct TcpContext {
  std::shared_ptr<Listener> listen(std::string_view addr);
  std::shared_ptr<Connection> connect(std::string_view addr);
};

struct Listener {
  Socket socket;
  Listener(Socket socket) : socket(std::move(socket)) {}

  void close() {
    socket.close();
  }

  void accept(Function<void(Error*, std::shared_ptr<Connection>)> callback);

  std::string localAddress() const;
};

struct Connection : std::enable_shared_from_this<Connection> {
  Socket socket;
  int readState = 0;
  Function<void(Error*, BufferHandle)> readCallback;
  std::array<char, 32> tmpReadBuffer;
  std::vector<size_t> bufferSizes;
  BufferHandle buffer;
  std::vector<iovec> iovecs;
  std::vector<Allocator> allocators;
  
  Connection(Socket socket) : socket(std::move(socket)) {}
  ~Connection();

  void close();
  void read(Function<void(Error*, BufferHandle)>);
  template<typename Buffer>
  void write(Buffer buffer, Function<void(Error*)>);

  std::string localAddress() const;
  std::string remoteAddress() const;

  int getFd() const {
    return socket.nativeFd();
  }
};

} // namespace ipc

} // namespace rpc
