/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

struct ConnectionImpl;
struct Connection : std::enable_shared_from_this<Connection> {
  Socket socket;

  Connection(Socket socket) : socket(std::move(socket)) {}
  ~Connection();

  void close();
  void read(Function<void(Error*, BufferHandle)>);
  template<typename Buffer>
  void write(Buffer buffer, Function<void(Error*)>);

  std::string localAddress() const;
  std::string remoteAddress() const;
};

} // namespace ipc

} // namespace rpc
