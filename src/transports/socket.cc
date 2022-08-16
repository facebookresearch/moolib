
#include "socket.h"

#include "rpc.h"
#include "intrusive_list.h"

#include "fmt/printf.h"

#include <stdexcept>

#include <limits.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <unistd.h>

#undef assert
#define assert(x) (bool(x) ? 0 : (printf("assert failure %s:%d\n", __FILE__, __LINE__), std::abort(), 0))


namespace rpc {

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

template<typename Duration>
static float seconds(Duration duration) {
  return std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(duration).count();
}


std::once_flag scheduleFlag;
void scheduleTest() {

  for (int i = 0; i != 10; ++i) {
    auto start = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    //fmt::printf("plain clock overhead is %f\n", seconds(now - start) * 1000);

    std::atomic_int doneCount = 0;
    constexpr int N = 1000;

    auto scheduleTime = std::chrono::steady_clock::now();
    for (int i = 0; i != N; ++i) {
      scheduler.run([&, scheduleTime]() {
        //std::this_thread::sleep_for(std::chrono::milliseconds(2));
        //fmt::printf("%d scheduled in %g\n", i, seconds(now - scheduleTime) * 1000);
        auto now = std::chrono::steady_clock::now();
        if (++doneCount == N) {
          float t = seconds(now - start);
          fmt::printf("everything done in %g (%g/s)\n", t * 1000, N / t);
        }
      });
    }

    now = std::chrono::steady_clock::now();
    float t = seconds(now - start);
    fmt::printf("run took %g (%g/s)\n", t * 1000, N / t);

    //while (doneCount < N) std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  std::exit(0);

}

template<typename F>
std::shared_ptr<ResolveHandle> resolveIpAddress(std::string_view address, int port, bool asynchronous, F&& callback) {
  auto h = std::make_shared<ResolveHandle>();
  h->address = address;
  h->service = std::to_string(port);
  h->req.ar_name = h->address.c_str();
  h->req.ar_service = h->service.c_str();
  h->req.ar_request = nullptr;

  h->callback = std::move(callback);

  //std::call_once(scheduleFlag, scheduleTest);

  Function<void()> f = [asynchronous, wh = std::weak_ptr<ResolveHandle>(h)]() {
    auto h = wh.lock();
    if (h) {
      int e = gai_error(&h->req);
      if (e == 0) {
        h->callback(nullptr, h->req.ar_result);
      } else {
        if (!asynchronous && e == EAI_ADDRFAMILY) {
          throw std::system_error(EAFNOSUPPORT, std::generic_category());
        }
        std::string str = e == EAI_SYSTEM ? std::strerror(errno) : gai_strerror(e);
        Error e(std::move(str));
        h->callback(&e, nullptr);
      }
    } else fmt::printf("h is null\n");
  };

  memset(&h->sevp, 0, sizeof(h->sevp));
  h->sevp.sigev_notify = asynchronous ? SIGEV_THREAD : SIGEV_NONE;
  h->sevp.sigev_value.sival_ptr = f.release();
  h->sevp.sigev_notify_function = [](sigval v) { Function<void()>((FunctionPointer)v.sival_ptr)(); };
  gaicb* ptr = &h->req;
  int r;
  do {
    r = getaddrinfo_a(asynchronous ? GAI_NOWAIT : GAI_WAIT, &ptr, 1, &h->sevp);
    //fmt::printf("getaddrinfo_a returned %d\n", r);
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

static std::pair<std::string_view, int> decodeIpAddress(std::string_view address) {
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

template<typename T>
struct WriteData {
  static constexpr bool isAllocated = true;
  moolib::IntrusiveListLink<WriteData> link;
  T* beginptr = nullptr;
  T* endptr = nullptr;
  T* dataptr = nullptr;
  size_t msize = 0;
  T* data() {
    return dataptr;
  }
  size_t size() const {
    return msize;
  }
  size_t capacity() const {
    return endptr - dataptr;
  }
  bool empty() const {
    return msize == 0;
  }

  template<typename... Args>
  void emplace_back(Args&&... args) {
    assert(capacity() > msize);
    new (data() + msize) T(std::forward<Args>(args)...);
    ++msize;
  }

  template<typename V>
  void push_back(V&& v) {
    emplace_back(std::forward<V>(v));
  }

  void erase(T* begin, T* end) {
    if (begin == dataptr) {
      for (auto* i = begin; i != end; ++i) {
        i->~T();
      }
      dataptr = end;
      msize -= end - begin;
      if (msize == 0) {
        dataptr = beginptr;
      }
    } else {
      throw std::invalid_argument("WriteData::erase");
    }
  }

  T* begin() {
    return dataptr;
  }
  T* end() {
    return dataptr + msize;
  }

  void clear() {
    erase(begin(), end());
    msize = 0;
    dataptr = beginptr;
  }

  static WriteData* create(size_t capacity) {
    WriteData* r = (WriteData*)std::aligned_alloc(std::max(alignof(T), alignof(WriteData)), sizeof(WriteData) + sizeof(T) * capacity);
    if (!r) {
      throw std::bad_alloc();
    }
    new (r) WriteData();
    r->beginptr = (T*)(r + 1);
    r->endptr = r->beginptr + capacity;
    r->dataptr = r->beginptr;
    return r;
  }
  static void destroy(WriteData* ptr) {
    ptr->clear();
    ptr->~WriteData();
    std::free(ptr);
  }
};
template<typename T>
struct WriteDataRef {
  static constexpr bool isAllocated = false;
  T* ptr = nullptr;
  size_t msize = 0;
  T* data() {
    return ptr;
  }
  size_t size() const {
    return msize;
  }
  static void destroy(WriteDataRef*) {}
};

template<typename T>
struct WriteDataHandle {
  static constexpr bool isAllocated = T::isAllocated;
  T* ptr = nullptr;
  WriteDataHandle() = default;
  WriteDataHandle(std::nullptr_t) {}
  WriteDataHandle(T* ptr) : ptr(ptr) {}
  WriteDataHandle(const WriteDataHandle&) = delete;
  WriteDataHandle(WriteDataHandle&& n) {
    std::swap(ptr, n.ptr);
  }
  WriteDataHandle& operator=(const WriteDataHandle&) = delete;
  WriteDataHandle& operator=(WriteDataHandle&& n) {
    std::swap(ptr, n.ptr);
    return *this;
  }
  ~WriteDataHandle() {
    if (ptr) {
      T::destroy(ptr);
    }
  }
  T& operator*() {
    return *ptr;
  }
  T* operator->() {
    return ptr;
  }
  T* release() {
    return std::exchange(ptr, nullptr);
  }
  explicit operator bool() const {
    return ptr != nullptr;
  }
};

struct SocketImpl : std::enable_shared_from_this<SocketImpl> {
  int af = -1;
  int fd = -1;
  std::atomic_bool closed = false;
  std::shared_ptr<ResolveHandle> resolveHandle;
  bool addedInPoll = false;

  using WriteCallback = std::pair<size_t, Function<void(Error*)>>;

  using WriteHandle = WriteDataHandle<WriteData<iovec>>;
  using WriteCallbackHandle = WriteDataHandle<WriteData<WriteCallback>>;

  alignas(64) std::atomic_int writeTriggerCount = 0;
  SpinMutex writeQueueMutex;
  moolib::IntrusiveList<WriteData<iovec>, &WriteData<iovec>::link> queuedWrites;
  moolib::IntrusiveList<WriteData<WriteCallback>, &WriteData<WriteCallback>::link> queuedWriteCallbacks;
  // std::vector<iovec> queuedWrites;
  // std::vector<std::pair<size_t, Function<void(Error*)>>> queuedWriteCallbacks;
  SpinMutex writeMutex;
  // moolib::IntrusiveList<WriteData<iovec>> newWrites;
  // moolib::IntrusiveList<WriteData<WriteCallback>> newWriteCallbacks;
  // moolib::IntrusiveList<WriteData<iovec>> activeWrite;
  // moolib::IntrusiveList<WriteData<WriteCallback>> activeWriteCallbacks;
  // std::vector<iovec> newWrites;
  // std::vector<std::pair<size_t, Function<void(Error*)>>> newWriteCallbacks;
  // std::vector<iovec> activeWrites;
  // std::vector<std::pair<size_t, Function<void(Error*)>>> activeWriteCallbacks;

  moolib::IntrusiveList<WriteData<iovec>, &WriteData<iovec>::link> freelistWrites;
  moolib::IntrusiveList<WriteData<WriteCallback>, &WriteData<WriteCallback>::link> freelistWriteCallbacks;

  alignas(64) std::atomic_int readTriggerCount = 0;
  bool wantsRead = false;
  SpinMutex readMutex;
  Function<void(Error*, std::unique_lock<SpinMutex>*)> onRead;
  std::vector<char> readBuffer;
  size_t readBufferOffset = 0;
  size_t readBufferFilled = 0;
  const void* prevReadPtr = nullptr;
  size_t prevReadOffset = 0;
  std::vector<int> receivedFds;

  template<typename T>
  void reallyClearList(T& list) {
    while (!list.empty()) {
      auto& v = list.back();
      list.pop_back();
      v.destroy(&v);
    }
  }

  // template<typename T>
  // void clearList(moolib::IntrusiveList<T>& list) {
  //   auto& freelist = std::is_same_v<T, iovec> ? freelistWrites : freelistWriteCallbacks;
  //   while (freelist.size() < 16 && !list.empty()) {
  //     auto& v = list.back();
  //     v.clear();
  //     list.pop_back();
  //     freelist.push_back(v);
  //   }
  //   reallyClearList(list);
  // }

  void close() {
    if (closed.exchange(true)) {
      return;
    }

    std::unique_lock l(readMutex, std::defer_lock);
    if (inReadLoop != this) {
      l.lock();
    }
    onRead = nullptr;
    std::lock_guard l2(writeMutex);
    std::unique_lock l3(writeQueueMutex);
    reallyClearList(queuedWrites);
    reallyClearList(queuedWriteCallbacks);
    if (resolveHandle) {
      resolveHandle = nullptr;
    }
    if (fd != -1) {
      //fmt::printf("close fd %d\n", fd);
      if (::close(fd)) {
        perror("close");
      }
      fd = -1;
    }
    for (auto v : receivedFds) {
      ::close(v);
    }
  }

  ~SocketImpl() {
    close();
    reallyClearList(freelistWrites);
    reallyClearList(freelistWriteCallbacks);
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
        throw std::system_error(errno, std::generic_category());
      }
      if (::listen(fd, 50) == -1) {
        throw std::system_error(errno, std::generic_category());
      }
      fmt::printf("unix socket %d listening on %s\n", fd, path);
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
        fmt::printf("unix socket %d connect to %s failed with error %s\n", fd, path, e.what());
        std::move(callback)(&e);
      } else {
        fmt::printf("unix socket %d connect to %s succeeded\n", fd, path);
        std::move(callback)(nullptr);
        std::unique_lock ql(writeQueueMutex);
        writeLoop(wl, ql);
      }
    } else {
      wl.unlock();
      int port = 0;
      //fmt::printf("connect to %s ?\n", address);
      std::tie(address, port) = decodeIpAddress(address);
      auto h = resolveIpAddress(
          address, port, true,
          [this, me = shared_from_this(), address = std::string(address)](Error* e, addrinfo* aix) {
            std::unique_lock wl(writeMutex);
            std::unique_lock rl(readMutex);
            if (closed.load(std::memory_order_relaxed)) {
              return;
            }
            //if (e) fmt::printf("resolve error %s\n", e->what());
            if (!e) {
              //fmt::printf("resolve ok\n");
              for (auto* i = aix; i; i = i->ai_next) {
                if ((i->ai_family == AF_INET || i->ai_family == AF_INET6) && i->ai_socktype == SOCK_STREAM) {
                  if (fd == -1) {
                    fd = ::socket(i->ai_family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
                    if (fd == -1) {
                      continue;
                    }
                  }

                  //fmt::printf("connect fd %d\n", fd);

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
      inReadLoop = this;
      while (true) {
        readTriggerCount.store(-0xffff, std::memory_order_relaxed);
        wantsRead = true;
        while (onRead && wantsRead) {
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
      WriteHandle wh(&queuedWrites.front());
      WriteCallbackHandle wch(&queuedWriteCallbacks.front());
      queuedWrites.pop_front();
      queuedWriteCallbacks.pop_front();
      ql.unlock();
      bool canWrite = writevImpl(std::move(wh), std::move(wch), ql);
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

  bool read(void* dst, size_t size) {
    if (closed.load(std::memory_order_relaxed) || fd == -1) {
      wantsRead = false;
      return false;
    }
    if (readBuffer.empty()) {
      readBuffer.resize(4096);
    }
    if (prevReadPtr != dst) {
      prevReadPtr = dst;
      prevReadOffset = 0;
      if (readBufferFilled - readBufferOffset > 0) {
        size_t n = std::min(size, readBufferFilled - readBufferOffset);
        std::memcpy(dst, readBuffer.data() + readBufferOffset, n);
        readBufferOffset += n;
        if (readBufferOffset == readBufferFilled) {
          readBufferOffset = 0;
          readBufferFilled = 0;
        }
        if (n >= size) {
          prevReadPtr = nullptr;
          return true;
        }
        prevReadOffset = n;
      }
      readBufferOffset = 0;
      readBufferFilled = 0;
    }
    std::array<iovec, 2> readIovecs;
    readIovecs[0].iov_base = (char*)dst + prevReadOffset;
    readIovecs[0].iov_len = size - prevReadOffset;
    readIovecs[1].iov_base = readBuffer.data() + readBufferOffset;
    readIovecs[1].iov_len = readBuffer.size() - readBufferFilled;
    if (readIovecs[0].iov_len == 0 || readIovecs[0].iov_len > size || readIovecs[1].iov_len == 0 ||
        readIovecs[1].iov_len > readBuffer.size()) {
      throw std::runtime_error("read bad iovec");
    }
    msghdr msg = {0};
    union {
      char buf[CMSG_SPACE(sizeof(int))];
      cmsghdr align;
    } u;
    msg.msg_iov = (::iovec*)readIovecs.data();
    msg.msg_iovlen = readIovecs.size();
    msg.msg_control = u.buf;
    msg.msg_controllen = sizeof(u.buf);
    readTriggerCount.store(-0xffff, std::memory_order_relaxed);
    //fmt::printf("read fd %d\n", fd);
    ssize_t r = ::recvmsg(fd, &msg, MSG_CMSG_CLOEXEC);
    if (r > 200) {
      //fmt::printf("recvmsg %d returned %d/%d\n", fd, r, readIovecs[0].iov_len + readIovecs[1].iov_len);
    }
    wantsRead = r == readIovecs[0].iov_len + readIovecs[1].iov_len;
    if (r == -1) {
      int error = errno;
      //fmt::printf("recvmsg error %s\n", std::strerror(error));
      //fmt::printf("error is %d\n", error);
      if (error == EINTR) {
        wantsRead = true;
        return false;
      }
      if (error == EAGAIN || error == EWOULDBLOCK || error == ENOTCONN) {
        return false;
      }
      Error e(std::strerror(error));
      //fmt::printf("read error %s\n", e.what());
      if (onRead) {
        onRead(&e, nullptr);
      }
      return false;
    } else {
      if (r == 0) {
        Error e("Connection closed");
        if (onRead) {
          onRead(&e, nullptr);
        }
        return false;
      }
      if (msg.msg_controllen != 0) {
        cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
          int fd;
          std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
          receivedFds.push_back(fd);
        }
      }
      size_t dstlen = size - prevReadOffset;
      prevReadOffset += std::min(dstlen, (size_t)r);
      if (r > dstlen) {
        readBufferFilled += r - dstlen;
      }
      if (prevReadOffset == size) {
        prevReadPtr = nullptr;
        return true;
      }
      return false;
    }
  }

  bool readv(const iovec* vec, size_t veclen) {
    if (closed.load(std::memory_order_relaxed) || fd == -1) {
      wantsRead = false;
      return false;
    }
    size_t skip = readBufferFilled - readBufferOffset;
    const iovec* ovec = vec;
    size_t oveclen = veclen;
    iovec iovecbackup;
    bool restoreIovecBackup = false;
    if (prevReadPtr != vec) {
      prevReadPtr = vec;
      prevReadOffset = 0;
    }
    if (skip) {
      size_t left = skip;
      for (size_t i = 0;; ++i) {
        if (i == veclen) {
          if (readBufferOffset == readBufferFilled) {
            readBufferOffset = 0;
            readBufferFilled = 0;
          }
          prevReadPtr = nullptr;
          return true;
        }
        if (left == 0) {
          break;
        }
        size_t n = std::min(left, vec[i].iov_len);
        std::memcpy(vec[i].iov_base, readBuffer.data() + readBufferOffset, n);
        readBufferOffset += n;
        left -= n;
        prevReadOffset += n;
        if (n < vec[i].iov_len) {
          break;
        }
      }
      if (readBufferOffset == readBufferFilled) {
        readBufferOffset = 0;
        readBufferFilled = 0;
      }
    }

    size_t left = prevReadOffset;
    if (left) {
      for (size_t i = 0;; ++i) {
        if (i == veclen) {
          throw std::runtime_error("readv skipped entire buffer");
        }
        if (left < vec[i].iov_len) {
          vec = &vec[i];
          veclen -= i;
          iovecbackup = *vec;
          restoreIovecBackup = true;
          iovec* v = (iovec*)vec;
          v[0].iov_base = (char*)v[0].iov_base + left;
          v[0].iov_len -= left;
          break;
        } else {
          left -= vec[i].iov_len;
        }
      }
    }

    ssize_t bytes = 0;
    for (size_t i = 0; i != veclen; ++i) {
      bytes += vec[i].iov_len;
    }

    readTriggerCount.store(-0xffff, std::memory_order_relaxed);
    ssize_t r = ::readv(fd, (::iovec*)vec, veclen);
    //fmt::printf("readv %d returned %d/%d bytes\n", fd, r, bytes);
    if (restoreIovecBackup) {
      *(iovec*)vec = iovecbackup;
      vec = ovec;
      veclen = oveclen;
    }
    wantsRead = r == bytes;
    if (r == -1) {
      int error = errno;
      //fmt::printf("readv error %s\n", std::strerror(error));
      if (error == EINTR) {
        wantsRead = true;
        return false;
      }
      if (error == EAGAIN || error == EWOULDBLOCK || error == ENOTCONN) {
        return false;
      }
      Error e(std::strerror(errno));
      if (onRead) {
        onRead(&e, nullptr);
      }
      return false;
    } else {
      if (r == 0) {
        Error e("Connection closed");
        if (onRead) {
          onRead(&e, nullptr);
        }
        return false;
      }
      if (r < bytes) {
        prevReadOffset += r;
        return false;
      } else {
        prevReadPtr = nullptr;
        return true;
      }
    }
  }

  template<typename WriteHandleT, typename WriteCallbackHandleT>
  bool writevImpl(WriteHandleT writes, WriteCallbackHandleT writeCallbacks, std::unique_lock<SpinMutex>& ql) {
    if (closed.load(std::memory_order_relaxed)) {
      return false;
    }
    TIME(writevImpl);

    const iovec* vec = writes->data();
    size_t veclen = writes->size();
    std::pair<size_t, Function<void(Error*)>>* callbacks = writeCallbacks->data();
    size_t callbacksLen = writeCallbacks->size();

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
    // size_t bytes = 0;
    // for (size_t i = 0; i != veclen; ++i) {
    //   bytes += vec[i].iov_len;
    // }
    //fmt::printf("write to fd %d\n", fd);
    writeTriggerCount.store(-0xffff, std::memory_order_relaxed);
    TIME(sendmsg);
    ssize_t r = ::sendmsg(fd, &msg, MSG_NOSIGNAL);
    ENDTIME(sendmsg);
    // if (r < bytes) {
    //   fmt::printf("write to fd %d returned %d/%d\n",fd, r, bytes);
    // }
    bool canWrite = false;
    if (r == -1) {
      int e = errno;
      if (e == EAGAIN || e == EWOULDBLOCK) {
        r = 0;
      }
    }
    if (r == -1) {
      int e = errno;
      if (e == EINTR) {
        canWrite = true;
      }
      Error ee(std::strerror(e));
      std::move(callbacks[0].second)(&ee);
      ql.lock();
      reallyClearList(queuedWrites);
      reallyClearList(queuedWriteCallbacks);
    } else {
      size_t writtenForCallback = r;
      WriteCallbackHandle qWriteCallbacks = nullptr;
      while (writtenForCallback) {
        if (callbacksLen == 0) {
          fmt::printf("writev empty callback list");
          ql.lock();
          assert(!queuedWriteCallbacks.empty());
          if constexpr (WriteCallbackHandleT::isAllocated) {
            writeCallbacks->clear();
            freelistWriteCallbacks.push_back(*writeCallbacks.release());
          }
          qWriteCallbacks = WriteDataHandle(&queuedWriteCallbacks.front());
          queuedWriteCallbacks.pop_front();
          ql.unlock();
          callbacks = qWriteCallbacks->data();
          callbacksLen = qWriteCallbacks->size();
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
      if (offset == msg.msg_iovlen) {
        canWrite = true;
      }
      //   ql.lock();
      //   writes->clear();
      //   writeCallbacks->clear();
      //   freelistWrites.push_back(*writes.release());
      //   fmt::printf("freelistWrites.size() is %d\n", std::distance(freelistWrites.begin(), freelistWrites.end()));
      //     freelistWriteCallbacks.push_back(*writeCallbacks.release());
      //   fmt::printf("freelistWriteCallbacks.size() is %d\n", std::distance(freelistWriteCallbacks.begin(), freelistWriteCallbacks.end()));
      // } else {
      if constexpr (WriteHandleT::isAllocated) {
        fmt::printf("erase vec %d (size %d)\n", offset, writes->size());
        writes->erase(writes->begin(), writes->begin() + offset);
        assert(writes->size() > 0);
        if (r) {
          assert(false);
          assert(!writes->empty());
          iovec& v = writes->data()[0];
          v.iov_base = (char*)v.iov_base + r;
          v.iov_len = v.iov_len - r;
        }
        ql.lock();
        if (writes->empty()) {
          freelistWrites.push_back(*writes.release());
          fmt::printf("freelistWrites.size() is %d\n", std::distance(freelistWrites.begin(), freelistWrites.end()));
        } else {
          queuedWrites.push_front(*writes.release());
        }
      } else {
        fmt::printf("copy vec %d\n", veclen - offset);
        if (offset != veclen) {
          WriteHandle newWrites(WriteData<iovec>::create(std::max((size_t)4096, veclen - offset)));
          for (size_t i = offset; i != veclen; ++i) {
            if (i == offset) {
              iovec v;
              v.iov_base = (char*)vec[i].iov_base + r;
              v.iov_len = vec[i].iov_len - r;
              newWrites->push_back(v);
            } else {
              newWrites->push_back(vec[i]);
            }
          }
          ql.lock();
          queuedWrites.push_front(*newWrites.release());
        }
      }
      if constexpr (WriteCallbackHandleT::isAllocated) {
        qWriteCallbacks = std::move(writeCallbacks);
      }
      if (qWriteCallbacks) {
        fmt::printf("erase q callbacks %d / %d\n", callbacks - qWriteCallbacks->data(), qWriteCallbacks->size());
        qWriteCallbacks->erase(qWriteCallbacks->begin(), qWriteCallbacks->begin() + (callbacks - qWriteCallbacks->data()));
        assert(qWriteCallbacks->size() == callbacksLen);
        if (callbacksLen) {
          qWriteCallbacks->data()[0].first -= writtenForCallback;
        }
        if (!ql.owns_lock()) {
          ql.lock();
        }
        if (qWriteCallbacks->empty()) {
          freelistWriteCallbacks.push_back(*qWriteCallbacks.release());
          fmt::printf("freelistWriteCallbacks.size() is %d\n", std::distance(freelistWriteCallbacks.begin(), freelistWriteCallbacks.end()));
        } else {
          queuedWriteCallbacks.push_front(*qWriteCallbacks.release());
        }
      } else {
        fmt::printf("copy %d callbacks\n", callbacksLen);
        WriteCallbackHandle newWriteCallbacks(WriteData<WriteCallback>::create(std::max((size_t)4096, callbacksLen)));
        for (size_t i = 0; i != callbacksLen; ++i) {
          if (i == 0) {
            newWriteCallbacks->emplace_back(callbacks[i].first - writtenForCallback, std::move(callbacks[i].second));
          } else {
            newWriteCallbacks->push_back(std::move(callbacks[i]));
          }
        }
        if (!ql.owns_lock()) {
          ql.lock();
        }
        queuedWriteCallbacks.push_front(*newWriteCallbacks.release());
      }
    }

    return canWrite;
  }

  void writev(const iovec* vec, size_t veclen, Function<void(Error*)> callback) {
    TIME(writev);
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
        WriteDataRef<const iovec> write;
        write.ptr = vec;
        write.msize = veclen;
        WriteDataRef<WriteCallback> writeCallback;
        writeCallback.ptr = &p;
        writeCallback.msize = 1;
        bool canWrite = writevImpl(WriteDataHandle(&write), WriteDataHandle(&writeCallback), ql);
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
    WriteData<iovec>* writeData = nullptr;
    if (!queuedWrites.empty()) {
      writeData = &queuedWrites.back();
    }
    if (!writeData || writeData->capacity() - writeData->size() < veclen) {
      writeData = WriteData<iovec>::create(std::max((size_t)4096, veclen));
      queuedWrites.push_back(*writeData);
    }
    size_t bytes = 0;
    for (size_t i = 0; i != veclen; ++i) {
      writeData->emplace_back(vec[i]);
      bytes += vec[i].iov_len;
    }
    WriteData<WriteCallback>* writeCallback = nullptr;
    if (!queuedWriteCallbacks.empty()) {
      writeCallback = &queuedWriteCallbacks.back();
    }
    if (!writeCallback || writeCallback->size() == writeCallback->capacity()) {
      writeCallback = WriteData<WriteCallback>::create(4096);
      queuedWriteCallbacks.push_back(*writeCallback);
    }
    writeCallback->emplace_back(bytes, std::move(callback));
    // bytes = 0;
    // for (auto& v : queuedWrites) {
    //   bytes += v.iov_len;
    // }
    //fmt::printf("tid %d, fd %d queued, queuedWrites.size() is now %d, %d bytes\n", ::gettid(), fd, queuedWrites.size(), bytes);
  }

  void sendFd(int fd, Function<void(Error*)> callback) {
    std::array<iovec, 2> iovecs;
    iovecs[0].iov_base = &writeFdFlag;
    iovecs[0].iov_len = 4;
    iovecs[1].iov_base = (void*)(uintptr_t)fd;
    iovecs[1].iov_len = 0;
    writev(iovecs.data(), 2, std::move(callback));
  }

  int recvFd() {
    uint32_t flag = 0;
    if (!read(&flag, sizeof(flag))) {
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

  std::string localAddress() const {
    if (af == AF_INET) {
      sockaddr_storage addr;
      socklen_t addrlen = sizeof(addr);
      if (::getsockname(fd, (sockaddr*)&addr, &addrlen)) {
        throw std::system_error(errno, std::generic_category());
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
        throw std::system_error(errno, std::generic_category());
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
        throw std::system_error(errno, std::generic_category());
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
            //fmt::printf("poll & trigger fd %d\n", impl->fd);
            impl->triggerWrite();
          } else {
            //fmt::printf("poll w/o trigger %d\n", impl->fd);
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
        throw std::system_error(errno, std::generic_category());
      }
      thread = std::thread([&] { entry(); });
    });
    epoll_event e;
    e.data.ptr = &*impl;
    e.events = EPOLLIN | EPOLLOUT | EPOLLET;
    if (epoll_ctl(fd, EPOLL_CTL_ADD, impl->fd, &e)) {
      throw std::system_error(errno, std::generic_category());
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
    throw std::system_error(errno, std::generic_category());
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

bool Socket::read(void* dst, size_t size) {
  return impl->read(dst, size);
}

bool Socket::readv(const iovec* vec, size_t veclen) {
  return impl->readv(vec, veclen);
}

void Socket::sendFd(int fd, Function<void(Error*)> callback) {
  impl->sendFd(fd, std::move(callback));
}

int Socket::recvFd() {
  return impl->recvFd();
}

std::string Socket::localAddress() const {
  return impl->localAddress();
}

std::string Socket::remoteAddress() const {
  return impl->remoteAddress();
}

} // namespace rpc
