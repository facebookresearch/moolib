/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "socket.h"

#include "rpc.h"
#include "vector.h"

#include "fmt/printf.h"

#include <array>
#include <atomic>
#include <stdexcept>
#include <system_error>

#include <ifaddrs.h>
#include <limits.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <type_traits>
#include <unistd.h>

namespace rpc {

using moolib::Vector;

namespace poll {
void add(std::shared_ptr<SocketImpl> impl);
}

struct ResolveHandle {
  gaicb req;
  sigevent sevp;
  std::string address;
  std::string service;
  Function<void(Error*, addrinfo*)> callback;

  ResolveHandle() = default;
  ResolveHandle(const ResolveHandle&) = delete;
  ResolveHandle& operator=(const ResolveHandle&) = delete;
  ~ResolveHandle() {
    gai_cancel(&req);
  }
};

template<typename F>
std::shared_ptr<ResolveHandle> resolveIpAddress(std::string_view address, int port, bool asynchronous, F&& callback) {
  auto h = std::make_shared<ResolveHandle>();
  h->address = address;
  h->service = std::to_string(port);
  h->req.ar_name = h->address.c_str();
  h->req.ar_service = h->service.c_str();
  h->req.ar_request = nullptr;

  h->callback = std::move(callback);

  Function<void()> f = [asynchronous, wh = std::weak_ptr<ResolveHandle>(h)]() {
    auto h = wh.lock();
    if (h) {
      int e = gai_error(&h->req);
      if (e == 0) {
        h->callback(nullptr, h->req.ar_result);
      } else {
        if (!asynchronous && e == EAI_ADDRFAMILY) {
          throw std::system_error(EAFNOSUPPORT, std::generic_category(), "getaddrinfo_a");
        }
        std::string str = e == EAI_SYSTEM ? std::strerror(errno) : gai_strerror(e);
        Error e(std::move(str));
        h->callback(&e, nullptr);
      }
    }
  };

  memset(&h->sevp, 0, sizeof(h->sevp));
  h->sevp.sigev_notify = asynchronous ? SIGEV_THREAD : SIGEV_NONE;
  h->sevp.sigev_value.sival_ptr = f.release();
  h->sevp.sigev_notify_function = [](sigval v) { Function<void()>((FunctionPointer)v.sival_ptr)(); };
  gaicb* ptr = &h->req;
  int r;
  do {
    r = getaddrinfo_a(asynchronous ? GAI_NOWAIT : GAI_WAIT, &ptr, 1, &h->sevp);
  } while (r == EAI_INTR);
  if (r) {
    Function<void()>{(FunctionPointer)h->sevp.sigev_value.sival_ptr};
    std::string str = r == EAI_SYSTEM ? std::strerror(errno) : gai_strerror(r);
    Error e(std::move(str));
    if (asynchronous) {
      scheduler.run([e = std::move(e), h]() mutable { h->callback(&e, nullptr); });
    } else {
      h->callback(&e, nullptr);
    }
    return nullptr;
  }
  if (!asynchronous) {
    Function<void()>{(FunctionPointer)h->sevp.sigev_value.sival_ptr}();
  }
  return h;
}

std::pair<std::string_view, int> decodeIpAddress(std::string_view address) {
  std::string_view hostname = address;
  int port = 0;
  auto bpos = address.find('[');
  if (bpos != std::string_view::npos) {
    auto bepos = address.find(']', bpos);
    if (bepos != std::string_view::npos) {
      hostname = address.substr(bpos + 1, bepos - (bpos + 1));
      address = address.substr(bepos + 1);
    }
  }
  auto cpos = address.find(':');
  if (cpos != std::string_view::npos) {
    if (hostname == address) {
      hostname = address.substr(0, cpos);
    }
    ++cpos;
    while (cpos != address.size()) {
      char c = address[cpos];
      if (c < '0' || c > '9') {
        break;
      }
      port *= 10;
      port += c - '0';
      ++cpos;
    }
  }
  return {hostname, port};
}

uint32_t writeFdFlag = 0x413ffc3f;

thread_local SocketImpl* inReadLoop = nullptr;

struct SocketImpl : std::enable_shared_from_this<SocketImpl> {
  int af = -1;
  int fd = -1;
  std::atomic_bool closed = false;
  bool readLoopClose = false;
  std::shared_ptr<ResolveHandle> resolveHandle;
  bool addedInPoll = false;

  alignas(64) std::atomic_int writeTriggerCount = 0;
  SpinMutex writeQueueMutex;
  Vector<iovec> queuedWrites;
  Vector<std::pair<size_t, Function<void(Error*)>>> queuedWriteCallbacks;
  SpinMutex writeMutex;
  Vector<iovec> newWrites;
  Vector<std::pair<size_t, Function<void(Error*)>>> newWriteCallbacks;
  Vector<iovec> activeWrites;
  Vector<std::pair<size_t, Function<void(Error*)>>> activeWriteCallbacks;

  alignas(64) std::atomic_int readTriggerCount = 0;
  bool wantsRead = false;
  SpinMutex readMutex;
  Function<void(Error*, std::unique_lock<SpinMutex>*)> onRead;
  std::vector<int> receivedFds;

  void closeImpl() {
    if (inReadLoop == this) {
      readLoopClose = true;
      return;
    }
    std::unique_lock l(readMutex);
    onRead = nullptr;
    std::lock_guard l2(writeMutex);
    std::unique_lock l3(writeQueueMutex);
    queuedWrites.clear();
    queuedWriteCallbacks.clear();
    if (resolveHandle) {
      resolveHandle = nullptr;
    }
    if (fd != -1) {
      ::close(fd);
      fd = -1;
    }
    for (auto v : receivedFds) {
      ::close(v);
    }
  }

  void close() {
    if (closed.exchange(true)) {
      return;
    }

    closeImpl();
  }

  ~SocketImpl() {
    close();
  }

  void listen(std::string_view address) {
    std::unique_lock wl(writeMutex);
    if (closed.load(std::memory_order_relaxed)) {
      return;
    }
    if (af == AF_UNIX) {
      sockaddr_un sa;
      memset(&sa, 0, sizeof(sa));
      sa.sun_family = AF_UNIX;
      sa.sun_path[0] = 0;
      std::string path = "moolib-" + std::string(address);
      size_t len = std::min(path.size(), sizeof(sa.sun_path) - 2);
      std::memcpy(&sa.sun_path[1], path.data(), len);
      if (::bind(fd, (const sockaddr*)&sa, sizeof(sa)) == -1) {
        throw std::system_error(errno, std::generic_category(), "bind");
      }
      if (::listen(fd, 50) == -1) {
        throw std::system_error(errno, std::generic_category(), "listen:");
      }
    } else if (af == AF_INET) {
      wl.unlock();
      int port = 0;
      std::tie(address, port) = decodeIpAddress(address);
      auto h =
          resolveIpAddress(address, port, false, [this, address = std::string(address), port](Error* e, addrinfo* aix) {
            if (e) {
              throw *e;
            } else {
              std::unique_lock wl(writeMutex);
              std::unique_lock rl(readMutex);
              std::string errors;
              int tries = 0;
              for (auto* i = aix; i; i = i->ai_next) {
                if ((i->ai_family == AF_INET || i->ai_family == AF_INET6) && i->ai_socktype == SOCK_STREAM) {
                  ++tries;

                  if (fd == -1) {
                    fd = ::socket(i->ai_family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
                    if (fd == -1) {
                      errors += fmt::sprintf("socket: %s", std::strerror(errno));
                      continue;
                    }
                  }

                  int reuseaddr = 1;
                  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuseaddr, sizeof(reuseaddr));
                  int reuseport = 1;
                  ::setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &reuseport, sizeof(reuseport));

                  if (::bind(fd, i->ai_addr, i->ai_addrlen) == 0 && ::listen(fd, 50) == 0) {
                    poll::add(shared_from_this());
                    return;
                  } else {
                    const char* se = std::strerror(errno);
                    ::close(fd);
                    fd = -1;

                    if (!errors.empty()) {
                      errors += ", ";
                    }
                    char buf[128];
                    memset(buf, 0, sizeof(buf));
                    int r = getnameinfo(i->ai_addr, i->ai_addrlen, buf, sizeof(buf) - 1, nullptr, 0, NI_NUMERICHOST);
                    const char* s = r ? gai_strerror(r) : buf;
                    errors += fmt::sprintf("%s (port %d): %s", s, port, se);
                  }
                }
              }
              if (tries == 0) {
                errors += "Name did not resolve to any usable addresses";
              }
              throw Error(std::move(errors));
            }
          });
      wl.lock();
      resolveHandle = std::move(h);
    } else {
      throw Error("listen: unkown address family\n");
    }
  }

  void setTcpSockOpts() {
    int nodelay = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
  }

  void accept(Function<void(Error*, Socket)> callback) {
    std::lock_guard l(readMutex);
    if (closed.load(std::memory_order_relaxed)) {
      return;
    }
    onRead = [this, callback = std::move(callback)](Error* error, auto* lock) mutable {
      while (true) {
        if (closed.load(std::memory_order_relaxed)) {
          wantsRead = false;
          return;
        }
        readTriggerCount.store(-0xffff, std::memory_order_relaxed);
        int r = ::accept4(fd, nullptr, 0, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (r == -1) {
          wantsRead = false;
          if (errno == EAGAIN) {
            return;
          }
          return;
        } else {
          Socket s;
          s.impl = std::make_shared<SocketImpl>();
          s.impl->af = af;
          s.impl->fd = r;
          s.impl->setTcpSockOpts();
          poll::add(s.impl);
          callback(nullptr, std::move(s));
        }
      }
    };
  }

  void connect(std::string_view address, Function<void(Error*)> callback) {
    std::unique_lock wl(writeMutex);
    if (closed.load(std::memory_order_relaxed)) {
      return;
    }
    if (af == AF_UNIX) {
      sockaddr_un sa;
      memset(&sa, 0, sizeof(sa));
      sa.sun_family = AF_UNIX;
      sa.sun_path[0] = 0;
      std::string path = "moolib-" + std::string(address);
      size_t len = std::min(path.size(), sizeof(sa.sun_path) - 2);
      std::memcpy(&sa.sun_path[1], path.data(), len);
      if (::connect(fd, (const sockaddr*)&sa, sizeof(sa)) && errno != EAGAIN) {
        Error e(std::strerror(errno));
        std::move(callback)(&e);
      } else {
        std::move(callback)(nullptr);
        std::unique_lock ql(writeQueueMutex);
        writeLoop(wl, ql);
      }
    } else {
      wl.unlock();
      int port = 0;
      std::tie(address, port) = decodeIpAddress(address);
      auto h = resolveIpAddress(
          address, port, true,
          [this, me = shared_from_this(), address = std::string(address)](Error* e, addrinfo* aix) {
            std::unique_lock wl(writeMutex);
            std::unique_lock rl(readMutex);
            if (closed.load(std::memory_order_relaxed)) {
              return;
            }
            if (!e) {
              for (auto* i = aix; i; i = i->ai_next) {
                if ((i->ai_family == AF_INET || i->ai_family == AF_INET6) && i->ai_socktype == SOCK_STREAM) {
                  if (fd == -1) {
                    fd = ::socket(i->ai_family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
                    if (fd == -1) {
                      continue;
                    }
                  }

                  if (::connect(fd, i->ai_addr, i->ai_addrlen) == 0 || errno == EAGAIN || errno == EINPROGRESS) {
                    rl.unlock();
                    setTcpSockOpts();
                    poll::add(shared_from_this());
                    std::unique_lock ql(writeQueueMutex);
                    writeLoop(wl, ql);
                    return;
                  } else {
                    ::close(fd);
                    fd = -1;
                  }
                }
              }
            }
          });
      wl.lock();
      resolveHandle = std::move(h);
    }
  }

  void triggerRead() {
    scheduler.run([me = shared_from_this(), this] {
      std::unique_lock l(readMutex);
      if (closed.load(std::memory_order_relaxed)) {
        return;
      }
      inReadLoop = this;
      while (true) {
        readTriggerCount.store(-0xffff, std::memory_order_relaxed);
        wantsRead = true;
        while (onRead && wantsRead) {
          wantsRead = false;
          onRead(nullptr, &l);
          if (!l.owns_lock()) {
            l.lock();
          }
        }
        int v = -0xffff;
        if (readTriggerCount.compare_exchange_strong(v, 0)) {
          break;
        }
      }
      inReadLoop = nullptr;
      l.unlock();
      if (readLoopClose) {
        closeImpl();
      }
    });
  }

  void triggerWrite() {
    scheduler.run([me = shared_from_this(), this] {
      std::unique_lock wl(writeMutex, std::try_to_lock);
      if (wl.owns_lock()) {
        std::unique_lock ql(writeQueueMutex);
        writeLoop(wl, ql);
      }
    });
  }

  void writeLoop(std::unique_lock<SpinMutex>& wl, std::unique_lock<SpinMutex>& ql) {
    while (true) {
      writeTriggerCount.store(-0xffff, std::memory_order_relaxed);
      if (queuedWrites.empty()) {
        wl.unlock();
        ql.unlock();
        return;
      }
      activeWrites.clear();
      activeWriteCallbacks.clear();
      std::swap(activeWrites, queuedWrites);
      std::swap(activeWriteCallbacks, queuedWriteCallbacks);
      ql.unlock();
      bool canWrite = writevImpl(
          activeWrites.data(), activeWrites.size(), activeWriteCallbacks.data(), activeWriteCallbacks.size(), ql);
      if (!ql.owns_lock()) {
        ql.lock();
      }
      if (queuedWrites.empty()) {
        wl.unlock();
        ql.unlock();
        return;
      }
      if (!canWrite) {
        wl.unlock();
        int v = -0xffff;
        if (writeTriggerCount.compare_exchange_strong(v, 0)) {
          ql.unlock();
          return;
        }
        wl.try_lock();
        if (!wl.owns_lock()) {
          ql.unlock();
          return;
        }
      }
    }
  }

  void setOnRead(Function<void(Error*, std::unique_lock<SpinMutex>* lock)> callback) {
    std::unique_lock l(readMutex);
    if (closed.load(std::memory_order_relaxed)) {
      return;
    }
    if (onRead) {
      throw Error("onRead callback is already set");
    }
    onRead = std::move(callback);
    l.unlock();
    triggerRead();
  }

  size_t readv(const iovec* vec, size_t veclen) {
    if (closed.load(std::memory_order_relaxed) || fd == -1) {
      wantsRead = false;
      return 0;
    }
    msghdr msg = {0};
    union {
      char buf[CMSG_SPACE(sizeof(int))];
      cmsghdr align;
    } u;
    msg.msg_control = u.buf;
    msg.msg_controllen = sizeof(u.buf);
    msg.msg_iov = (::iovec*)vec;
    msg.msg_iovlen = std::min(veclen, (size_t)IOV_MAX);
    readTriggerCount.store(-0xffff, std::memory_order_relaxed);
    ssize_t r = ::recvmsg(fd, &msg, 0);
    wantsRead = true;
    if (r == -1) {
      int error = errno;
      if (error == EINTR) {
        return 0;
      }
      wantsRead = false;
      if (error == EAGAIN || error == EWOULDBLOCK || error == ENOTCONN) {
        return 0;
      }
      Error e(std::strerror(error));
      if (onRead) {
        onRead(&e, nullptr);
      }
      return 0;
    } else {
      if (r == 0) {
        Error e("Connection closed");
        if (onRead) {
          onRead(&e, nullptr);
        }
        return 0;
      }
      if (msg.msg_controllen != 0) {
        cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
          int fd;
          std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
          receivedFds.push_back(fd);
        }
      }
      return r;
    }
  }

  bool writevImpl(
      const iovec* vec, size_t veclen, std::pair<size_t, Function<void(Error*)>>* callbacks, size_t callbacksLen,
      std::unique_lock<SpinMutex>& ql) {
    if (closed.load(std::memory_order_relaxed)) {
      return false;
    }

    msghdr msg;
    union {
      char buf[CMSG_SPACE(sizeof(int))];
      cmsghdr align;
    } u;

    msg.msg_name = nullptr;
    msg.msg_namelen = 0;
    msg.msg_control = nullptr;
    msg.msg_controllen = 0;
    msg.msg_flags = 0;

    static_assert(sizeof(iovec) == sizeof(::iovec));
    msg.msg_iov = (::iovec*)vec;
    msg.msg_iovlen = std::min(veclen, (size_t)IOV_MAX);
    if (af == AF_UNIX) {
      for (size_t i = 0; i != msg.msg_iovlen; ++i) {
        if (vec[i].iov_base == (void*)&writeFdFlag) {
          if (i == 0) {
            int fd = (int)(uintptr_t)vec[i + 1].iov_base;
            msg.msg_iovlen = 1;
            msg.msg_control = u.buf;
            msg.msg_controllen = sizeof(u.buf);
            cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
            cmsg->cmsg_level = SOL_SOCKET;
            cmsg->cmsg_type = SCM_RIGHTS;
            cmsg->cmsg_len = CMSG_LEN(sizeof(fd));
            std::memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));
            break;
          } else {
            msg.msg_iovlen = i;
            break;
          }
        }
      }
    }
    writeTriggerCount.store(-0xffff, std::memory_order_relaxed);
    ssize_t r = ::sendmsg(fd, &msg, MSG_NOSIGNAL);
    bool canWrite = true;
    if (r == -1) {
      int e = errno;
      if (e == EAGAIN || e == EWOULDBLOCK) {
        canWrite = false;
        r = 0;
      } else if (e == EINTR) {
        r = 0;
      }
    }
    if (r == -1) {
      canWrite = false;
      int e = errno;
      Error ee(std::strerror(e));
      std::move(callbacks[0].second)(&ee);
      activeWrites.clear();
      activeWriteCallbacks.clear();
      ql.lock();
      queuedWrites.clear();
      queuedWriteCallbacks.clear();
    } else {
      size_t writtenForCallback = r;
      while (writtenForCallback) {
        if (callbacksLen == 0) {
          throw Error("writev empty callback list");
        }
        if (writtenForCallback >= callbacks[0].first) {
          writtenForCallback -= callbacks[0].first;
          ++callbacks;
          --callbacksLen;
        } else {
          break;
        }
      }
      size_t offset = 0;
      for (; offset != veclen; ++offset) {
        if (r < vec[offset].iov_len) {
          break;
        }
        r -= vec[offset].iov_len;
      }
      if (offset == veclen) {
        activeWrites.clear();
        activeWriteCallbacks.clear();
      } else {
        newWrites.clear();
        newWriteCallbacks.clear();
        if (vec == activeWrites.data() && veclen == activeWrites.size()) {
          activeWrites.erase(activeWrites.begin(), activeWrites.begin() + offset);
          if (r) {
            iovec& v = activeWrites[0];
            v.iov_base = (char*)v.iov_base + r;
            v.iov_len = v.iov_len - r;
          }
          std::swap(newWrites, activeWrites);
        } else {
          for (size_t i = offset; i != veclen; ++i) {
            if (i == offset) {
              iovec v;
              v.iov_base = (char*)vec[i].iov_base + r;
              v.iov_len = vec[i].iov_len - r;
              newWrites.push_back(v);
            } else {
              newWrites.push_back(vec[i]);
            }
          }
        }
        if (callbacks + callbacksLen == activeWriteCallbacks.data() + activeWriteCallbacks.size()) {
          activeWriteCallbacks.erase(
              activeWriteCallbacks.begin(), activeWriteCallbacks.begin() + (callbacks - activeWriteCallbacks.data()));
          if (writtenForCallback) {
            activeWriteCallbacks[0].first -= writtenForCallback;
          }
          std::swap(newWriteCallbacks, activeWriteCallbacks);
        } else {
          for (size_t i = 0; i != callbacksLen; ++i) {
            if (i == 0) {
              newWriteCallbacks.emplace_back(callbacks[i].first - writtenForCallback, std::move(callbacks[i].second));
            } else {
              newWriteCallbacks.push_back(std::move(callbacks[i]));
            }
          }
        }
        activeWrites.clear();
        activeWriteCallbacks.clear();
        ql.lock();
        if (!queuedWrites.empty()) {
          for (auto& v : queuedWrites) {
            newWrites.push_back(v);
          }
          for (auto& v : queuedWriteCallbacks) {
            newWriteCallbacks.push_back(std::move(v));
          }
          queuedWrites.clear();
          queuedWriteCallbacks.clear();
        }
        std::swap(queuedWrites, newWrites);
        std::swap(queuedWriteCallbacks, newWriteCallbacks);
      }
    }

    return canWrite;
  }

  void writev(const iovec* vec, size_t veclen, Function<void(Error*)> callback) {
    std::unique_lock ql(writeQueueMutex);
    if (closed.load(std::memory_order_relaxed)) {
      return;
    }
    std::unique_lock wl(writeMutex, std::defer_lock);
    if (queuedWrites.empty()) {
      if (wl.try_lock() && fd != -1) {
        ql.unlock();
        size_t bytes = 0;
        for (size_t i = 0; i != veclen; ++i) {
          bytes += vec[i].iov_len;
        }
        std::pair<size_t, Function<void(Error*)>> p;
        p.first = bytes;
        p.second = std::move(callback);
        bool canWrite = writevImpl(vec, veclen, &p, 1, ql);
        if (!ql.owns_lock()) {
          ql.lock();
        }
        if (queuedWrites.empty()) {
          wl.unlock();
          ql.unlock();
          return;
        }
        if (!canWrite) {
          wl.unlock();
          int v = -0xffff;
          if (writeTriggerCount.compare_exchange_strong(v, 0)) {
            ql.unlock();
            return;
          }
          wl.try_lock();
          if (!wl.owns_lock()) {
            ql.unlock();
            return;
          }
        }
        writeLoop(wl, ql);
        return;
      } else if (wl.owns_lock()) {
        wl.unlock();
      }
    }
    size_t bytes = 0;
    for (size_t i = 0; i != veclen; ++i) {
      queuedWrites.push_back(vec[i]);
      bytes += vec[i].iov_len;
    }
    queuedWriteCallbacks.emplace_back(bytes, std::move(callback));
  }

  void sendFd(int fd, Function<void(Error*)> callback) {
    std::array<iovec, 2> iovecs;
    iovecs[0].iov_base = &writeFdFlag;
    iovecs[0].iov_len = 4;
    iovecs[1].iov_base = (void*)(uintptr_t)fd;
    iovecs[1].iov_len = 0;
    writev(iovecs.data(), 2, std::move(callback));
  }

  int recvFd(CachedReader& reader) {
    uint32_t flag = 0;
    if (!reader.readCopy(&flag, sizeof(flag))) {
      return -1;
    }
    if (flag != writeFdFlag) {
      Error e("recvFd flag mismatch");
      if (onRead) {
        onRead(&e, nullptr);
      }
      return -1;
    }
    if (receivedFds.empty()) {
      Error e("receivedFds is empty!");
      if (onRead) {
        onRead(&e, nullptr);
      }
      return -1;
    }
    int fd = receivedFds.front();
    receivedFds.erase(receivedFds.begin());
    return fd;
  }

  static std::string ipAndPort(const sockaddr* addr, socklen_t addrlen) {
    char host[128];
    memset(host, 0, sizeof(host));
    char port[16];
    memset(port, 0, sizeof(port));
    int r = getnameinfo(addr, addrlen, host, sizeof(host) - 1, port, sizeof(port) - 1, NI_NUMERICHOST | NI_NUMERICSERV);
    if (r) {
      throw std::runtime_error(gai_strerror(r));
    }
    if (addr->sa_family == AF_INET) {
      return fmt::sprintf("%s:%s", host, port);
    } else if (addr->sa_family == AF_INET6) {
      return fmt::sprintf("[%s]:%s", host, port);
    } else {
      return "";
    }
  }

  std::vector<std::string> localAddresses() const {
    if (af == AF_INET) {
      std::vector<std::string> r;
      sockaddr_storage addr;
      socklen_t addrlen = sizeof(addr);
      if (::getsockname(fd, (sockaddr*)&addr, &addrlen)) {
        throw std::system_error(errno, std::generic_category(), "getsockname");
      }
      bool isAnyAddr = false;
      int port = 0;
      if (addr.ss_family == AF_INET) {
        const sockaddr_in* sa = (const sockaddr_in*)&addr;
        port = sa->sin_port;
        if (sa->sin_addr.s_addr == INADDR_ANY) {
          isAnyAddr = true;
        }
      } else if (addr.ss_family == AF_INET6) {
        const sockaddr_in6* sa = (const sockaddr_in6*)&addr;
        port = sa->sin6_port;
        if (!memcmp(&sa->sin6_addr, &in6addr_any, sizeof(in6addr_any))) {
          isAnyAddr = true;
        }
      }
      if (isAnyAddr) {
        struct ifaddrs* list;
        if (::getifaddrs(&list) == 0) {
          for (; list; list = list->ifa_next) {
            try {
              if (list->ifa_addr && (list->ifa_flags & IFF_RUNNING) == IFF_RUNNING &&
                  list->ifa_addr->sa_family == addr.ss_family) {
                if (addr.ss_family == AF_INET) {
                  sockaddr_in sa;
                  std::memcpy(&sa, list->ifa_addr, sizeof(sa));
                  sa.sin_port = port;
                  r.push_back(ipAndPort((const sockaddr*)&sa, sizeof(sa)));
                } else if (addr.ss_family == AF_INET6) {
                  sockaddr_in6 sa;
                  std::memcpy(&sa, list->ifa_addr, sizeof(sa));
                  sa.sin6_port = port;
                  r.push_back(ipAndPort((const sockaddr*)&sa, sizeof(sa)));
                }
              }
            } catch (const std::exception& e) {
            }
          }
          ::freeifaddrs(list);
        }
      }
      return r;
    } else {
      return {};
    }
  }

  std::string localAddress() const {
    if (af == AF_INET) {
      sockaddr_storage addr;
      socklen_t addrlen = sizeof(addr);
      if (::getsockname(fd, (sockaddr*)&addr, &addrlen)) {
        throw std::system_error(errno, std::generic_category(), "getsockname");
      }
      return ipAndPort((const sockaddr*)&addr, addrlen);
    } else {
      return "";
    }
  }

  std::string remoteAddress() const {
    if (af == AF_INET) {
      sockaddr_storage addr;
      socklen_t addrlen = sizeof(addr);
      if (::getpeername(fd, (sockaddr*)&addr, &addrlen)) {
        throw std::system_error(errno, std::generic_category(), "getpeername");
      }
      return ipAndPort((const sockaddr*)&addr, addrlen);
    } else {
      return "";
    }
  }
};

namespace poll {

struct PollThread {
  ~PollThread() {
    terminate = true;
    if (thread.joinable()) {
      thread.join();
    }
  }
  bool terminate = false;
  std::once_flag flag;
  std::thread thread;
  int fd = -1;
  std::atomic_bool anyDead = false;
  std::mutex mutex;
  std::vector<std::shared_ptr<SocketImpl>> activeList;
  std::vector<std::shared_ptr<SocketImpl>> deadList;

  void entry() {
    std::array<epoll_event, 1024> events;

    while (!terminate) {
      int n = epoll_wait(fd, events.data(), events.size(), 250);
      if (n < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw std::system_error(errno, std::generic_category(), "epoll_wait");
      }
      for (int i = 0; i != n; ++i) {
        if (events[i].events & (EPOLLIN | EPOLLERR)) {
          SocketImpl* impl = (SocketImpl*)events[i].data.ptr;
          if (++impl->readTriggerCount == 1) {
            impl->triggerRead();
          }
        }
        if (events[i].events & EPOLLOUT) {
          SocketImpl* impl = (SocketImpl*)events[i].data.ptr;
          if (++impl->writeTriggerCount == 1) {
            impl->triggerWrite();
          }
        }
      }

      if (n < events.size() && anyDead.load(std::memory_order_relaxed)) {
        anyDead = false;
        std::lock_guard l(mutex);
        deadList.clear();
      }
    }

    close(fd);
  }

  void add(std::shared_ptr<SocketImpl> impl) {
    std::lock_guard l(mutex);
    std::call_once(flag, [&] {
      fd = epoll_create1(EPOLL_CLOEXEC);
      if (fd == -1) {
        throw std::system_error(errno, std::generic_category(), "epoll_create1");
      }
      thread = std::thread([&] { entry(); });
    });
    epoll_event e;
    e.data.ptr = &*impl;
    e.events = EPOLLIN | EPOLLOUT | EPOLLET;
    if (epoll_ctl(fd, EPOLL_CTL_ADD, impl->fd, &e)) {
      throw std::system_error(errno, std::generic_category(), "epoll_ctl");
    }
    impl->addedInPoll = true;
    activeList.push_back(std::move(impl));
  }
  void remove(SocketImpl* impl) {
    std::lock_guard l(mutex);
    impl->addedInPoll = false;
    epoll_event e;
    epoll_ctl(fd, EPOLL_CTL_DEL, impl->fd, &e);

    for (auto i = activeList.begin(); i != activeList.end(); ++i) {
      if (impl == &**i) {
        anyDead = true;
        deadList.push_back(std::move(*i));
        activeList.erase(i);
        break;
      }
    }
  }
};
PollThread pollThread;

void add(std::shared_ptr<SocketImpl> impl) {
  pollThread.add(impl);
}

void remove(SocketImpl* impl) {
  pollThread.remove(impl);
}

} // namespace poll

Socket Socket::Unix() {
  Socket r;
  r.impl = std::make_shared<SocketImpl>();
  r.impl->af = AF_UNIX;
  r.impl->fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (r.impl->fd == -1) {
    throw std::system_error(errno, std::generic_category(), "socket");
  }
  poll::add(r.impl);
  return r;
}

Socket Socket::Tcp() {
  Socket r;
  r.impl = std::make_shared<SocketImpl>();
  r.impl->af = AF_INET;
  r.impl->fd = -1;
  return r;
}

Socket::Socket() {}
Socket::Socket(Socket&& n) {
  std::swap(impl, n.impl);
}
Socket& Socket::operator=(Socket&& n) {
  std::swap(impl, n.impl);
  return *this;
}

Socket::~Socket() {
  close();
}

void Socket::close() {
  if (impl) {
    if (impl->addedInPoll) {
      poll::remove(&*impl);
    }
    impl->close();
  }
}

void Socket::writev(const iovec* vec, size_t veclen, Function<void(Error*)> callback) {
  impl->writev(vec, veclen, std::move(callback));
}

void Socket::listen(std::string_view address) {
  impl->listen(address);
}

void Socket::accept(Function<void(Error*, Socket)> callback) {
  impl->accept(std::move(callback));
}

void Socket::connect(std::string_view address, Function<void(Error*)> callback) {
  impl->connect(address, std::move(callback));
}

void Socket::setOnRead(Function<void(Error*, std::unique_lock<SpinMutex>*)> callback) {
  impl->setOnRead(std::move(callback));
}

size_t Socket::readv(const iovec* vec, size_t veclen) {
  return impl->readv(vec, veclen);
}

void Socket::sendFd(int fd, Function<void(Error*)> callback) {
  impl->sendFd(fd, std::move(callback));
}

int Socket::recvFd(CachedReader& reader) {
  return impl->recvFd(reader);
}

std::vector<std::string> Socket::localAddresses() const {
  return impl->localAddresses();
}

std::string Socket::localAddress() const {
  return impl->localAddress();
}

std::string Socket::remoteAddress() const {
  return impl->remoteAddress();
}

int Socket::nativeFd() const {
  return impl->fd;
}

} // namespace rpc
