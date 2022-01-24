/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "rpc.h"

#include <tensorpipe/tensorpipe.h>

#include "fmt/printf.h"

#include <deque>
#include <list>
#include <new>
#include <random>
#include <thread>
#include <time.h>

namespace rpc {

async::SchedulerFifo scheduler;

std::mutex logMutex;

template<typename... Args>
void logimpl(const char* fmt, Args&&... args) {
  std::lock_guard l(logMutex);
  time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  struct tm tm;
  memset(&tm, 0, sizeof(tm));
  localtime_r(&now, &tm);
  char buf[0x40];
  std::strftime(buf, sizeof(buf), "%d-%m-%Y %H:%M:%S", &tm);
  auto s = fmt::sprintf(fmt, std::forward<Args>(args)...);
  if (!s.empty() && s.back() == '\n') {
    fmt::printf("%s: %s", buf, s);
  } else {
    fmt::printf("%s: %s\n", buf, s);
  }
  fflush(stdout);
}

template<typename... Args>
void log(const char* fmt, Args&&... args) {
  return;
  logimpl(fmt, std::forward<Args>(args)...);
}

template<typename... Args>
[[noreturn]] void fatal(const char* fmt, Args&&... args) {
  auto s = fmt::sprintf(fmt, std::forward<Args>(args)...);
  logimpl(" -- FATAL ERROR --\n%s\n", s);
  std::abort();
}

namespace {
auto seedRng() {
  std::random_device dev;
  auto start = std::chrono::high_resolution_clock::now();
  std::seed_seq ss(
      {(uint32_t)dev(), (uint32_t)dev(), (uint32_t)(std::chrono::high_resolution_clock::now() - start).count(),
       (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count(),
       (uint32_t)std::chrono::system_clock::now().time_since_epoch().count(),
       (uint32_t)std::chrono::high_resolution_clock::now().time_since_epoch().count(),
       (uint32_t)(std::chrono::high_resolution_clock::now() - start).count(), (uint32_t)dev(),
       (uint32_t)(std::chrono::high_resolution_clock::now() - start).count(), (uint32_t)dev(), (uint32_t)dev(),
       (uint32_t)std::hash<std::thread::id>()(std::this_thread::get_id())});
  return std::mt19937_64(ss);
};
thread_local std::mt19937_64 rng{seedRng()};
template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
T random(T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max()) {
  return std::uniform_int_distribution<T>(min, max)(rng);
}
} // namespace

std::string randomAddress() {
  std::string s;
  for (int i = 0; i != 2; ++i) {
    uint64_t v = random<uint64_t>();
    if (!s.empty()) {
      s += "-";
    }
    for (size_t i = 0; i != 8; ++i) {
      uint64_t sv = v >> (i * 8);
      s += "0123456789abcdef"[(sv >> 4) & 0xf];
      s += "0123456789abcdef"[sv & 0xf];
    }
  }
  return s;
}

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

struct TPSHMContext {
  tensorpipe_moorpc::Context& context = [] {
    static tensorpipe_moorpc::Context context = [] {
      tensorpipe_moorpc::Context context;
#ifndef __APPLE__
      context.registerTransport(0, "shm", tensorpipe_moorpc::transport::shm::create());
#endif
      context.registerChannel(0, "basic", tensorpipe_moorpc::channel::basic::create());
      return context;
    }();
    return std::ref(context);
  }();
  TPSHMContext() {
    // context.registerChannel(-1, "xth", tensorpipe_moorpc::channel::xth::create());
    // context.registerChannel(-1, "cuda_xth", tensorpipe_moorpc::channel::cuda_xth::create());

    // fixme? cuda_ipc gets deadlocked easily (test_multinode_allreduce)
    // context.registerChannel(-1, "cuda_ipc", tensorpipe_moorpc::channel::cuda_ipc::create());
  }
  auto listen(std::string_view addr) {
    return context.listen({"shm://" + std::string(addr)});
  }
  auto connect(std::string_view addr) {
    return context.connect("shm://" + std::string(addr));
  }
};

struct API_TPSHM {
  using Context = TPSHMContext;
  using Connection = std::shared_ptr<tensorpipe_moorpc::Pipe>;
  using Listener = std::shared_ptr<tensorpipe_moorpc::Listener>;

  static constexpr bool supportsBuffer = false;
  static constexpr bool persistentRead = false;
  static constexpr bool persistentAccept = false;
  static constexpr bool addressIsIp = false;
  static constexpr bool singularWrites = false;
  static constexpr bool supportsCuda = false;

  static std::vector<std::string> defaultAddr() {
    return {randomAddress()};
  }
  static std::string localAddr([[maybe_unused]] const Listener& listener, std::string addr) {
    return addr;
  }
  static std::string localAddr([[maybe_unused]] const Connection&) {
    return "";
  }
  static std::string remoteAddr([[maybe_unused]] const Connection&) {
    return "";
  }

  static auto& cast(Connection& x) {
    return *x;
  }
  static auto& cast(Listener& x) {
    return *x;
  }
  static std::string errstr(const tensorpipe_moorpc::Error& err) {
    return err.what();
  }
};

struct TPUVContext {
  tensorpipe_moorpc::Context& context = [] {
    static tensorpipe_moorpc::Context context = [] {
      tensorpipe_moorpc::Context context;
      context.registerTransport(0, "uv", tensorpipe_moorpc::transport::uv::create());
      context.registerChannel(0, "basic", tensorpipe_moorpc::channel::basic::create());
      return context;
    }();
    return std::ref(context);
  }();
  TPUVContext() {
    // context.registerChannel(-1, "cuda_xth", tensorpipe_moorpc::channel::cuda_xth::create());
  }
  auto listen(std::string_view addr) {
    return context.listen({"uv://" + std::string(addr)});
  }
  auto connect(std::string_view addr) {
    return context.connect("uv://" + std::string(addr));
  }
};

struct API_TPUV {
  using Context = TPUVContext;
  using Connection = std::shared_ptr<tensorpipe_moorpc::Pipe>;
  using Listener = std::shared_ptr<tensorpipe_moorpc::Listener>;

  static constexpr bool supportsBuffer = false;
  static constexpr bool persistentRead = false;
  static constexpr bool persistentAccept = false;
  static constexpr bool addressIsIp = true;
  static constexpr bool singularWrites = false;
  static constexpr bool supportsCuda = false;

  static std::vector<std::string> defaultAddr() {
    return {"0.0.0.0", "::"};
  }

  static std::string localAddr([[maybe_unused]] const Listener& listener, [[maybe_unused]] std::string addr) {
    for (auto& v : listener->addresses()) {
      return v.second;
    }
    return "";
  }
  static std::string localAddr([[maybe_unused]] const Connection& x) {
    return x->localAddr();
  }
  static std::string remoteAddr([[maybe_unused]] const Connection& x) {
    return x->remoteAddr();
  }

  static auto& cast(Connection& x) {
    return *x;
  }
  static auto& cast(Listener& x) {
    return *x;
  }
  static std::string errstr(const tensorpipe_moorpc::Error& err) {
    return err.what();
  }
};

struct TPIBVContext {
  tensorpipe_moorpc::Context& context = [] {
    static tensorpipe_moorpc::Context context = [] {
      tensorpipe_moorpc::Context context;
#ifndef __APPLE__
      context.registerTransport(0, "ibv", tensorpipe_moorpc::transport::ibv::create());
#endif
      context.registerChannel(0, "basic", tensorpipe_moorpc::channel::basic::create());
      return context;
    }();
    return std::ref(context);
  }();
  TPIBVContext() {
    // context.registerChannel(-1, "cuda_xth", tensorpipe_moorpc::channel::cuda_xth::create());
  }
  auto listen(std::string_view addr) {
    return context.listen({"ibv://" + std::string(addr)});
  }
  auto connect(std::string_view addr) {
    return context.connect("ibv://" + std::string(addr));
  }
};

struct API_TPIBV {
  using Context = TPIBVContext;
  using Connection = std::shared_ptr<tensorpipe_moorpc::Pipe>;
  using Listener = std::shared_ptr<tensorpipe_moorpc::Listener>;

  static constexpr bool supportsBuffer = false;
  static constexpr bool persistentRead = false;
  static constexpr bool persistentAccept = false;
  static constexpr bool addressIsIp = true;
  static constexpr bool singularWrites = false;
  static constexpr bool supportsCuda = false;

  static std::vector<std::string> defaultAddr() {
    return {"0.0.0.0", "::"};
  }

  static std::string localAddr([[maybe_unused]] const Listener& listener, [[maybe_unused]] std::string addr) {
    for (auto& v : listener->addresses()) {
      return v.second;
    }
    return "";
  }
  static std::string localAddr([[maybe_unused]] const Connection& x) {
    return "";
  }
  static std::string remoteAddr([[maybe_unused]] const Connection& x) {
    return "";
  }

  static auto& cast(Connection& x) {
    return *x;
  }
  static auto& cast(Listener& x) {
    return *x;
  }
  static std::string errstr(const tensorpipe_moorpc::Error& err) {
    return err.what();
  }
};

template<typename API>
struct APIWrapper : API {
  template<typename T, typename X = API>
  static auto errstr(T&& err) -> decltype(X::errstr(std::forward<T>(err))) {
    return API::errstr(std::forward<T>(err));
  }
  static std::string errstr(const char* str) {
    return std::string(str);
  }
  static std::string errstr(std::string_view str) {
    return std::string(str);
  }
  static std::string errstr(std::string&& str) {
    return std::move(str);
  }
};

enum class ConnectionType { uv, shm, ibv, count };
static const std::array<const char*, (int)ConnectionType::count> connectionTypeName = {
    "TCP/IP",
    "Shared memory",
    "InfiniBand",
};
static const std::array<const char*, (int)ConnectionType::count> connectionShortTypeName = {
    "uv",
    "shm",
    "ibv",
};
static const std::array<bool, (int)ConnectionType::count> connectionDefaultEnabled = {
    true,  // uv
    true,  // shm
    false, // ibv // disabled due to buggy
};

template<typename API>
struct index_t;
template<>
struct index_t<API_TPUV> {
  static constexpr ConnectionType value = ConnectionType::uv;
};
template<>
struct index_t<API_TPSHM> {
  static constexpr ConnectionType value = ConnectionType::shm;
};
template<>
struct index_t<API_TPIBV> {
  static constexpr ConnectionType value = ConnectionType::ibv;
};
template<typename API>
constexpr size_t index = (size_t)index_t<API>::value;

template<typename F>
auto switchOnAPI(ConnectionType t, F&& f) {
  switch (t) {
  case ConnectionType::uv:
    return f(API_TPUV{});
  case ConnectionType::shm:
    return f(API_TPSHM{});
  case ConnectionType::ibv:
    return f(API_TPIBV{});
  default:
    fatal("switchOnAPI bad index %d", (int)t);
  }
}

template<typename F>
auto switchOnScheme(std::string_view str, F&& f) {
  if (str == "uv") {
    return f(API_TPUV{});
  } else if (str == "shm") {
    return f(API_TPSHM{});
  } else if (str == "ibv") {
    return f(API_TPIBV{});
  } else {
    fatal("Unrecognized scheme '%s'", str);
  }
}

template<typename T>
struct RpcImpl;

bool addressIsIp(ConnectionType t) {
  return switchOnAPI(t, [](auto api) { return decltype(api)::addressIsIp; });
}

struct ConnectionTypeInfo {
  std::string_view name;
  std::vector<std::string_view> addr;

  template<typename X>
  void serialize(X& x) {
    x(name, addr);
  }
};

struct RpcConnectionImplBase {
  virtual ~RpcConnectionImplBase() {}
  virtual void close() = 0;

  virtual const std::string& localAddr() const = 0;
  virtual const std::string& remoteAddr() const = 0;
  virtual size_t apiIndex() const = 0;

  std::atomic_bool dead{false};
  std::atomic_int activeOps{0};
  std::atomic<std::chrono::steady_clock::time_point> lastReceivedData = std::chrono::steady_clock::time_point{};
  bool isExplicit = false;
  std::string connectAddr;
  std::chrono::steady_clock::time_point timeWait = std::chrono::steady_clock::time_point{};
};

struct RpcListenerImplBase {
  virtual ~RpcListenerImplBase() {}

  virtual void close() = 0;
  virtual std::string localAddr() const {
    return "";
  }

  std::atomic<bool> dead = false;
  std::atomic_int activeOps{0};
};

struct Connection {
  std::atomic<bool> valid = false;
  std::atomic<float> readBanditValue = 0.0f;
  std::atomic<std::chrono::steady_clock::time_point> lastTryConnect = std::chrono::steady_clock::time_point{};
  std::atomic<int> connectionAttempts = 0;
  std::atomic<std::chrono::steady_clock::time_point> lastRecv = std::chrono::steady_clock::time_point{};
  SpinMutex mutex;
  bool outgoing = false;
  std::string addr;
  bool isExplicit = false;
  std::atomic<bool> hasConn = false;
  std::vector<std::unique_ptr<RpcConnectionImplBase>> conns;

  std::chrono::steady_clock::time_point creationTimestamp = std::chrono::steady_clock::now();

  std::vector<std::string_view> remoteAddresses;

  alignas(64) SpinMutex latencyMutex;
  std::chrono::steady_clock::time_point lastUpdateLatency;
  std::atomic<float> runningLatency = 0.0f;
  float writeBanditValue = 0.0f;

  std::atomic<uint64_t> sendCount = 0;
};

struct Listener {
  int activeCount = 0;
  int explicitCount = 0;
  int implicitCount = 0;
  std::vector<std::unique_ptr<RpcListenerImplBase>> listeners;
};

struct PeerId {
  std::array<uint64_t, 2> id{};
  template<typename X>
  void serialize(X& x) {
    x(id);
  }

  bool operator==(const PeerId& n) const noexcept {
    return id == n.id;
  }
  bool operator!=(const PeerId& n) const noexcept {
    return id != n.id;
  }

  static PeerId generate() {
    return {random<uint64_t>(), random<uint64_t>()};
  }

  std::string toString() const {
    std::string s;
    for (auto v : id) {
      if (!s.empty()) {
        s += "-";
      }
      for (size_t i = 0; i != 8; ++i) {
        uint64_t sv = v >> (i * 8);
        s += "0123456789abcdef"[(sv >> 4) & 0xf];
        s += "0123456789abcdef"[sv & 0xf];
      }
    }
    return s;
  }
};

struct RemoteFunction {
  uint32_t id = 0;
  std::string_view typeId;

  template<typename X>
  void serialize(X& x) {
    x(id, typeId);
  }
};

template<typename API>
struct RpcConnectionImpl;

struct Deferrer {
  FunctionPointer head = nullptr;
  const bool call = true;
  Deferrer(bool call = true) : call(call) {}
  bool empty() const noexcept {
    return head == nullptr;
  }
  void operator()(Function<void()> func) {
    FunctionPointer o = head;
    head = func.release();
    head->next = o;
  }
  void execute() noexcept {
    if (call) {
      for (FunctionPointer i = head; i;) {
        FunctionPointer next = i->next;
        Function<void()>{i}();
        i = next;
      }
      head = nullptr;
    } else {
      for (FunctionPointer i = head; i;) {
        FunctionPointer next = i->next;
        Function<void()>{i};
        i = next;
      }
      head = nullptr;
    }
  }
  ~Deferrer() {
    execute();
  }
};

struct PeerImpl {
  Rpc::Impl& rpc;
  std::atomic_int activeOps{0};
  std::atomic_bool dead{false};

  alignas(64) SpinMutex idMutex_;
  std::atomic<bool> hasId = false;
  PeerId id;
  std::string_view name;

  std::array<Connection, (int)ConnectionType::count> connections_;

  alignas(64) SpinMutex remoteFuncsMutex_;
  std::unordered_map<std::string_view, RemoteFunction> remoteFuncs_;

  std::atomic<int> findThisPeerIncrementingTimeoutMilliseconds = 0;
  std::atomic<std::chrono::steady_clock::time_point> lastFindThisPeer = std::chrono::steady_clock::time_point{};
  static_assert(
      std::atomic<std::chrono::steady_clock::time_point>::is_always_lock_free, "atomic time_point is not lock-free");

  std::chrono::steady_clock::time_point lastFindPeers;

  struct RecentIncomingItem {
    uint32_t rid;
    std::chrono::steady_clock::time_point timeout;
  };

  alignas(64) SpinMutex recentIncomingMutex;
  std::deque<RecentIncomingItem> recentIncomingList;
  std::unordered_set<uint32_t> recentIncomingMap;

  PeerImpl(Rpc::Impl& rpc) : rpc(rpc) {}
  ~PeerImpl() {
    dead = true;
    while (activeOps.load()) {
      std::this_thread::yield();
    }
  }

  void clearRecentIncomingTimeouts() {
    auto now = std::chrono::steady_clock::now();
    while (!recentIncomingList.empty() && now >= recentIncomingList.front().timeout) {
      recentIncomingMap.erase(recentIncomingList.front().rid);
      recentIncomingList.pop_front();
    }
  }

  void addRecentIncoming(uint32_t rid, std::chrono::steady_clock::time_point timeout) {
    std::lock_guard ril(recentIncomingMutex);
    recentIncomingList.push_back({rid, timeout});
    recentIncomingMap.insert(rid);
  }

  void setRemoteFunc(std::string_view name, const RemoteFunction& rf) {
    std::lock_guard l(remoteFuncsMutex_);
    remoteFuncs_[name] = rf;
  }

  uint32_t functionId(std::string_view name) {
    std::lock_guard l(remoteFuncsMutex_);
    auto i = remoteFuncs_.find(name);
    if (i != remoteFuncs_.end()) {
      return i->second.id;
    } else {
      return 0;
    }
  }

  std::string_view functionName(uint32_t id) {
    std::lock_guard l(remoteFuncsMutex_);
    for (auto& v : remoteFuncs_) {
      if (v.second.id == id) {
        return v.first;
      }
    }
    return "<unknown>";
  }

  bool isConnected(const Connection& v) {
    return v.valid.load(std::memory_order_relaxed) && v.hasConn.load(std::memory_order_relaxed);
  }

  bool willConnectOrSend(const std::chrono::steady_clock::time_point& now, const Connection& v) {
    return v.valid.load(std::memory_order_relaxed) &&
           (v.hasConn.load(std::memory_order_relaxed) ||
            v.lastTryConnect.load(std::memory_order_relaxed) +
                    std::chrono::seconds(
                        v.connectionAttempts < 2 ||
                                now - v.lastRecv.load(std::memory_order_relaxed) <= std::chrono::seconds(10)
                            ? 2
                            : 30) <=
                now);
  }

  template<typename Buffer>
  bool banditSend(
      uint32_t mask, Buffer buffer, Deferrer& defer, size_t* indexUsed = nullptr,
      Me<RpcConnectionImplBase>* outConnection = nullptr, bool shouldFindPeer = true) noexcept {
    // log("banditSend %d bytes mask %#x\n", (int)buffer->size, mask);
    auto now = std::chrono::steady_clock::now();
    thread_local std::vector<std::pair<size_t, float>> list;
    list.clear();
    float sum = 0.0f;
    bool hasCudaTensor = false;
    auto* tensors = buffer->tensors();
    for (size_t i = 0; i != buffer->nTensors; ++i) {
      if (tensors[i].tensor.is_cuda()) {
        hasCudaTensor = true;
        break;
      }
    }
    for (size_t i = 0; i != connections_.size(); ++i) {
      if (~mask & (1 << i)) {
        continue;
      }
      if (hasCudaTensor) {
        fatal("CUDA tensors are currently not supported, sorry!");
        bool supportsCuda = false;
        switchOnAPI((ConnectionType)i, [&](auto api) { supportsCuda = decltype(api)::supportsCuda; });
        if (!supportsCuda) {
          continue;
        }
      }
      auto& v = connections_[i];
      if (willConnectOrSend(now, v)) {
        float score = std::exp(v.readBanditValue * 4);
        // log("bandit %s has score %g\n", connectionTypeName[i], score);
        sum += score;
        list.emplace_back(i, sum);
      }
    }
    if (list.size() > 0) {
      size_t index;
      if (list.size() == 1) {
        index = list[0].first;
      } else {
        float v = std::uniform_real_distribution<float>(0.0f, sum)(rng);
        index = std::lower_bound(list.begin(), std::prev(list.end()), v, [&](auto& a, float b) {
                  return a.second < b;
                })->first;
      }
      // log("bandit chose %d (%s)\n", index, connectionTypeName.at(index));
      auto& x = connections_.at(index);
      x.sendCount.fetch_add(1, std::memory_order_relaxed);
      bool b = switchOnAPI(
          (ConnectionType)index, [&](auto api) { return send<decltype(api)>(now, buffer, outConnection, defer); });
      if (!b && buffer) {
        mask &= ~(1 << index);
        return banditSend(mask, std::move(buffer), defer, indexUsed, outConnection, shouldFindPeer);
      }
      if (b && indexUsed) {
        *indexUsed = index;
      }
      return b;
    } else {
      // log("No connectivity to %s\n", name);

      if (shouldFindPeer) {
        int timeout = findThisPeerIncrementingTimeoutMilliseconds;
        if (now - lastFindThisPeer.load(std::memory_order_relaxed) >= std::chrono::milliseconds(timeout)) {
          log("findpeer timeout is %d\n", timeout);
          findThisPeerIncrementingTimeoutMilliseconds.store(
              std::min(std::max(timeout, 250) * 2, 1000), std::memory_order_relaxed);
          lastFindThisPeer.store(now, std::memory_order_relaxed);
          findPeer();
        }
      }
      return false;
    }
  }

  template<typename... Args>
  void log(const char* fmt, Args&&... args);

  std::string_view rpcName();

  void findPeer();

  template<typename API, bool isExplicit>
  void connect(std::string_view addr, Deferrer& defer);

  template<typename API, typename Buffer>
  bool send(
      std::chrono::steady_clock::time_point now, Buffer& buffer, Me<RpcConnectionImplBase>* outConnection,
      Deferrer& defer) {
    auto& x = connections_[index<API>];
    std::unique_lock l(x.mutex);
    if (x.conns.empty()) {
      if (x.connectionAttempts.load(std::memory_order_relaxed) == 0 ||
          now - x.lastTryConnect.load(std::memory_order_relaxed) >= std::chrono::seconds(1)) {
        x.lastTryConnect = now;
        ++x.connectionAttempts;
        if (x.remoteAddresses.empty()) {
          x.valid = false;
        } else {
          std::string_view addr;
          if (x.remoteAddresses.size() == 1) {
            addr = x.remoteAddresses[0];
          } else {
            addr = x.remoteAddresses[random<size_t>(0, x.remoteAddresses.size() - 1)];
          }
          l.unlock();
          if (!addr.empty()) {
            static std::atomic_int connects = 0;
            log("connecting to %s::%s!! :D total %d\n", connectionTypeName[index<API>], addr, ++connects);
            connect<API, false>(addr, defer);
          }
        }
      }
      if (outConnection) {
        *outConnection = nullptr;
      }
      return false;
    } else {
      size_t i = random<size_t>(0, x.conns.size() - 1);
      auto& c = x.conns[i];
      if (c->dead.load(std::memory_order_relaxed)) {
        log("Connection through %s to %s is dead, yo!\n", connectionTypeName[index<API>], name);
        BufferHandle buffer;
        serializeToBuffer(buffer, (uint32_t)0, (uint32_t)Rpc::reqClose);
        ((RpcConnectionImpl<API>&)*c).send(std::move(buffer), defer);
        throwAway(x, i);
        if (outConnection) {
          *outConnection = nullptr;
        }
        return false;
      } else {
        ((RpcConnectionImpl<API>&)*c).send(std::move(buffer), defer);
        if (outConnection) {
          *outConnection = makeMe(&*c);
        }
        return true;
      }
    }
  }

  void throwAway(Connection& x, size_t i) {
    auto cv = std::move(x.conns[i]);
    log("Throwing away connection %s to %s\n", connectionTypeName.at(cv->apiIndex()), name);
    std::swap(x.conns.back(), x.conns[i]);
    x.conns.pop_back();
    if (x.conns.empty()) {
      x.hasConn = false;
    }
    cv->timeWait = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    throwAway(std::move(cv));
  }

  void throwAway(std::unique_ptr<RpcConnectionImplBase> c);
};

std::string emptyString;

template<typename API>
struct RpcConnectionImpl : RpcConnectionImplBase {
  RpcConnectionImpl(RpcImpl<API>& rpc, typename API::Connection&& connection)
      : rpc(rpc), connection(std::move(connection)) {}
  RpcImpl<API>& rpc;

  typename API::Connection connection;

  PeerImpl* peer = nullptr;

  std::atomic_bool hasReceivedData{false};

  mutable std::once_flag localAddrOnce_;
  mutable std::once_flag remoteAddrOnce_;
  mutable std::string localAddrStr_;
  mutable std::string remoteAddrStr_;

  SpinMutex sendMutex;
  Buffer* sendQueueBegin = nullptr;
  Buffer* sendQueueEnd = nullptr;

  ~RpcConnectionImpl() {
    close();
    while (activeOps.load()) {
      std::this_thread::yield();
    }

    if constexpr (API::singularWrites) {
      for (Buffer* buf = sendQueueBegin; buf;) {
        SharedBufferHandle h;
        h.acquire(buf);
        buf = buf->next;
      }
    }
  }

  virtual const std::string& localAddr() const override {
    if (!hasReceivedData) {
      return emptyString;
    }
    std::call_once(localAddrOnce_, [this]() {
      try {
        localAddrStr_ = API::localAddr(connection);
      } catch (const std::exception&) {
      }
    });
    return localAddrStr_;
  }
  virtual const std::string& remoteAddr() const override {
    if (!hasReceivedData) {
      return emptyString;
    }
    std::call_once(remoteAddrOnce_, [this]() {
      try {
        remoteAddrStr_ = API::remoteAddr(connection);
      } catch (const std::exception&) {
      }
    });
    return remoteAddrStr_;
  }

  virtual size_t apiIndex() const override {
    return index<API>;
  }

  virtual void close() override {
    if (dead.exchange(true)) {
      return;
    }
    rpc.log("Connection %s closed\n", connectionTypeName[index<API>]);
    API::cast(connection).close();
  }

  void onError([[maybe_unused]] Error* err) {
    rpc.log("Connection %s to %s error: %s\n", connectionTypeName[index<API>], connectAddr, err->what());
    close();
  }
  void onError([[maybe_unused]] const char* err) {
    rpc.log("Connection %s error: %s\n", connectionTypeName[index<API>], err);
    close();
  }

  template<typename E>
  void onError(E&& error) {
    Error err(API::errstr(error));
    onError(&err);
  }

  static constexpr uint64_t kSignature = 0xff984b883019d446;

  void onData(BufferHandle buffer) noexcept {
    const std::byte* ptr = buffer->data();
    size_t len = buffer->size;
    rpc.log("%s :: got %d bytes\n", connectionTypeName[index<API>], len);
    if (len < sizeof(uint32_t) * 2) {
      onError("Received not enough data");
      return;
    }
    if (!hasReceivedData.load(std::memory_order_relaxed)) {
      hasReceivedData = true;
    }
    auto now = std::chrono::steady_clock::now();
    lastReceivedData.store(now, std::memory_order_relaxed);
    uint32_t rid;
    std::memcpy(&rid, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    len -= sizeof(uint32_t);
    uint32_t fid;
    std::memcpy(&fid, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    len -= sizeof(uint32_t);
    rpc.log("onData rid %#x fid %#x\n", rid, fid);
    if (peer && fid != Rpc::reqGreeting) {
      auto& x = peer->connections_.at(index<API>);
      x.lastRecv.store(now, std::memory_order_relaxed);
      Deferrer defer;
      if (rid & 1) {
        rpc.onRequest(*peer, *this, rid, fid, ptr, len, std::move(buffer), defer);
      } else {
        rpc.onResponse(*peer, *this, rid, fid, ptr, len, std::move(buffer), defer);
      }
    } else if (fid == Rpc::reqGreeting) {
      try {
        uint64_t signature;
        deserializeBufferPart(ptr, len, signature);
        if (signature == kSignature) {
          std::string_view peerName;
          PeerId peerId;
          std::vector<ConnectionTypeInfo> info;
          deserializeBuffer(ptr, len, signature, peerName, peerId, info);
          rpc.onGreeting(*this, peerName, peerId, std::move(info));
        } else {
          rpc.log("signature mismatch\n");
          close();
        }
      } catch (const std::exception&) {
        rpc.log("error in greeting\n");
        close();
      }
    }
  }

  void greet(std::string_view name, PeerId peerId, const std::vector<ConnectionTypeInfo>& info, Deferrer& defer) {
    log("%p::greet(\"%s\", %s)\n", (void*)this, std::string(name).c_str(), peerId.toString().c_str());
    BufferHandle buffer;
    serializeToBuffer(buffer, (uint32_t)0, (uint32_t)Rpc::reqGreeting, kSignature, name, peerId, info);
    send(std::move(buffer), defer);
  }

  void read(Me<RpcConnectionImpl>&& me) {
    rpc.log("read %s :: %p\n", connectionTypeName[index<API>], (void*)this);
    API::cast(connection)
        .readDescriptor([me = std::move(me)](auto&& error, tensorpipe_moorpc::Message msg) mutable noexcept {
          me->rpc.log("%s :: %p got data\n", connectionTypeName[index<API>], (void*)&*me);
          if (me->dead.load(std::memory_order_relaxed)) {
            me->rpc.log("already dead!\n");
            return;
          }
          if (error) {
            me->onError(error);
          } else {
            if (msg.metadata.size() != 4 || msg.tensors.size() == 0) {
              me->onError("Received Invalid data");
            } else {
              uint32_t size;
              deserializeBuffer(msg.metadata.data(), 4, size);
              // me->rpc.log("got %d bytes\n", size);
              BufferHandle buffer = makeBuffer(size, msg.tensors.size() - 1);
              bool valid = true;
              if (msg.tensors[0].buffer.type != tensorpipe_moorpc::DeviceType::kCpu) {
                valid = false;
              }
              msg.tensors[0].buffer.cpu.ptr = buffer->data();
              if (msg.tensors[0].buffer.cpu.length !=
                  size_t((std::byte*)(buffer->tensorMetaDataOffsets() + buffer->nTensors) - buffer->data())) {
                valid = false;
              }
              std::vector<Allocator> allocators;
              allocators.reserve(msg.tensors.size() - 1);
              for (size_t i = 1; i != msg.tensors.size(); ++i) {
                auto& tensor = msg.tensors[i];
                if (tensor.buffer.type == tensorpipe_moorpc::DeviceType::kCpu) {
                  allocators.emplace_back(rpc::kCPU, tensor.buffer.cpu.length);
                  tensor.buffer.cpu.ptr = allocators.back().data();
#ifdef USE_CUDA
                } else if (tensor.buffer.type == tensorpipe_moorpc::DeviceType::kCuda) {
                  allocators.emplace_back(rpc::kCUDA, tensor.buffer.cpu.length);
                  tensor.buffer.cpu.ptr = allocators.back().data();
#endif
                } else
                  me->onError("Received invalid tensor device type");
              }
              if (valid) {
                API::cast(me->connection)
                    .read(
                        std::move(msg),
                        [allocators = std::move(allocators), buffer = std::move(buffer), me = std::move(me)](
                            auto&& error, [[maybe_unused]] tensorpipe_moorpc::Message msg) mutable noexcept {
                          if (error) {
                            me->onError(error);
                          } else {
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
                              deserializeBufferPart(
                                  data + offsets[i], len > offsets[i] ? len - offsets[i] : 0, dtype, sizes, strides);
                              t = allocators[i].set(dtype, sizes, strides);
                            }
                            me->onData(std::move(buffer));

                            me->read(std::move(me));
                          }
                        });
              } else {
                me->onError("Received invalid data (2)");
              }
            }
          }
        });
  }

  void start(Deferrer& defer) {
    defer([me = makeMe(this)]() mutable { me->read(std::move(me)); });
  }

  template<typename Buffer>
  void send(Buffer buffer, Deferrer& defer) {
    // rpc.log("%s :: send %d bytes\n", connectionTypeName[index<API>], buffer->size);
    defer([buffer = std::move(buffer), me = makeMe(this)]() mutable {
      tensorpipe_moorpc::Message msg;
      msg.metadata.resize(4);
      if ((uint32_t)buffer->size != buffer->size) {
        fatal("send: buffer is too large (size does not fit in 32 bits)!");
      }
      serializeToStringView(std::string_view(msg.metadata), (uint32_t)buffer->size);
      msg.tensors.resize(buffer->nTensors + 1);
      msg.tensors[0].buffer.type = tensorpipe_moorpc::DeviceType::kCpu;
      msg.tensors[0].buffer.cpu.ptr = buffer->data();
      msg.tensors[0].buffer.cpu.length =
          (std::byte*)(buffer->tensorMetaDataOffsets() + buffer->nTensors) - buffer->data();
      auto* tensors = buffer->tensors();
      for (size_t i = 0; i != buffer->nTensors; ++i) {
        Tensor& tensor = tensors[i].tensor;
        auto& buf = msg.tensors[i + 1].buffer;
        if (tensor.is_cuda()) {
#ifdef USE_CUDA
          buf.type = tensorpipe_moorpc::DeviceType::kCuda;
          buf.cuda.ptr = tensor.data_ptr();
          buf.cuda.length = computeStorageNbytes(tensor.sizes(), tensor.strides(), tensor.itemsize());
#else
          fatal("Received CUDA tensor in non-CUDA build");
#endif
        } else {
          buf.type = tensorpipe_moorpc::DeviceType::kCpu;
          buf.cpu.ptr = tensor.data_ptr();
          buf.cpu.length = computeStorageNbytes(tensor.sizes(), tensor.strides(), tensor.itemsize());
        }
      }
      API::cast(me->connection)
          .write(
              std::move(msg), [buffer = std::move(buffer),
                               me = std::move(me)](auto&& error, [[maybe_unused]] tensorpipe_moorpc::Message msg) {
                if (error) {
                  me->onError(std::forward<decltype(error)>(error));
                }
              });
    });
  }
};

template<typename API>
struct RpcListenerImpl : RpcListenerImplBase {
  RpcListenerImpl(RpcImpl<API>& rpc, typename API::Listener&& listener, std::string_view addr)
      : rpc(rpc), listener(std::move(listener)), addr(addr) {
    accept();
  }
  ~RpcListenerImpl() {
    close();
    while (activeOps.load()) {
      std::this_thread::yield();
    }
  }
  RpcImpl<API>& rpc;
  typename API::Listener listener;
  bool isExplicit = false;
  bool active = false;
  std::string addr;

  virtual void close() override {
    if (dead.exchange(true, std::memory_order_relaxed)) {
      return;
    }
    rpc.log("Listener %s closed\n", connectionTypeName[index<API>]);
    dead = true;
    API::cast(listener).close();
  }

  virtual std::string localAddr() const override {
    return API::localAddr(listener, addr);
  }

  void accept() {
    API::cast(listener).accept([me = makeMe(this)](auto&& error, auto&& conn) mutable {
      if (error) {
        // Tensorpipe accept will report errors on some connection setup errors,
        // such as when connecting then immediately closing the connection.
        // As a result, we ignore errors and hope for the best.
        // Update: changed tensorpipe to ignore accept errors instead, otherwise
        // it would get stuck in an infinite accept/error loop here
        if (true) {
          if constexpr (std::is_same_v<std::decay_t<decltype(error)>, Error*>) {
            me->rpc.onAccept(*me, nullptr, error);
          } else {
            Error err(API::errstr(error));
            me->rpc.onAccept(*me, nullptr, &err);
          }
        } else {
          if (!API::persistentAccept && !me->dead) {
            me->accept();
          }
        }
      } else {
        auto c = std::make_unique<RpcConnectionImpl<API>>(me->rpc, std::move(conn));
        me->rpc.onAccept(*me, std::move(c), nullptr);
        if (!API::persistentAccept && !me->dead) {
          me->accept();
        }
      }
    });
  }
};

namespace {
template<typename T>
void listInsert(T* at, T* item) {
  T* next = at;
  T* prev = at->prev;
  next->prev = item;
  prev->next = item;
  item->next = next;
  item->prev = prev;
}
template<typename T>
void listErase(T* at) {
  T* next = at->next;
  T* prev = at->prev;
  next->prev = prev;
  prev->next = next;

  at->prev = nullptr;
  at->next = nullptr;
}
} // namespace

struct WeakLock {
  void* ptr = nullptr;
  std::atomic_int count = 1;
  WeakLock(void* ptr) : ptr(ptr) {}

  template<typename U>
  Me<U> lock() {
    int n = count.load(std::memory_order_relaxed);
    do {
      if (n == 0) {
        return nullptr;
      }
    } while (!count.compare_exchange_weak(n, n + 1));
    auto r = makeMe((U*)ptr);
    count.fetch_sub(1);
    return r;
  }
};

struct RpcImplBase {
  Rpc::Impl& rpc;
  std::atomic_int activeOps{0};
  std::shared_ptr<WeakLock> weakLock = std::make_shared<WeakLock>(this);
  RpcImplBase(Rpc::Impl& rpc) : rpc(rpc) {}
  virtual ~RpcImplBase() {
    --weakLock->count;
    while (weakLock->count.load()) {
      std::this_thread::yield();
    }
    while (activeOps.load()) {
      std::this_thread::yield();
    }
  }
};

struct Rpc::Impl {

  alignas(64) SpinMutex mutex_;
  std::list<std::string> stringList_;
  std::unordered_set<std::string_view> stringMap_;
  std::unordered_map<std::string_view, uint32_t> funcIds_;
  std::unordered_map<uint32_t, std::unique_ptr<Rpc::FBase>> funcs_;
  static constexpr size_t maxFunctions_ = 0x1000000;
  uint32_t baseFuncId_ = Rpc::reqCallOffset;

  struct Resend {
    SharedBufferHandle buffer;
    std::chrono::steady_clock::time_point ackTimestamp;
    std::chrono::steady_clock::time_point pokeTimestamp;
    std::chrono::steady_clock::time_point lastSendTimestamp;
    std::chrono::steady_clock::time_point lastSendFailTimestamp;
    bool hasAddedFailureLatency = false;
    size_t connectionIndex = ~0;
    Me<RpcConnectionImplBase> connection = nullptr;
    int pokeCount = 0;
    int totalPokeCount = 0;
    bool acked = false;
    int nackCount = 0;
  };

  struct Receive {
    BufferHandle buffer;
    bool done = false;
  };

  struct Incoming {
    Incoming* prev = nullptr;
    Incoming* next = nullptr;
    uint32_t rid;
    std::chrono::steady_clock::time_point responseTimestamp;
    PeerImpl* peer = nullptr;

    int timeoutCount = 0;
    std::chrono::steady_clock::time_point timeout;
    Receive recv;

    std::chrono::steady_clock::time_point creationTimestamp = std::chrono::steady_clock::now();
    Resend resend;
  };

  struct IncomingBucket {
    alignas(64) SpinMutex mutex;
    std::unordered_map<uint32_t, Incoming> map;
  };

  alignas(64) std::array<IncomingBucket, 0x10> incoming_;
  alignas(64) SpinMutex incomingFifoMutex_;
  Incoming incomingFifo_;
  std::atomic_size_t totalResponseSize_ = 0;

  std::thread timeoutThread_;
  alignas(64) std::once_flag timeoutThreadOnce_;
  Semaphore timeoutSem_;
  std::atomic<std::chrono::steady_clock::time_point> timeout_ = std::chrono::steady_clock::now();
  std::atomic<bool> terminate_ = false;

  std::atomic<int> timeoutMilliseconds_ = 1000 * 120;

  struct Outgoing {
    Outgoing* prev = nullptr;
    Outgoing* next = nullptr;
    std::chrono::steady_clock::time_point requestTimestamp;
    std::chrono::steady_clock::time_point timeout;
    uint32_t rid = 0;
    uint32_t fid = 0;
    Rpc::ResponseCallback response;
    PeerImpl* peer = nullptr;
    int timeoutCount = 0;
    std::chrono::steady_clock::time_point creationTimestamp = std::chrono::steady_clock::now();
    Resend resend;
    Receive recv;
  };

  struct OutgoingBucket {
    alignas(64) SpinMutex mutex;
    std::unordered_map<uint32_t, Outgoing> map;
  };

  alignas(64) std::array<OutgoingBucket, 0x10> outgoing_;
  alignas(64) SpinMutex outgoingFifoMutex_;
  Incoming outgoingFifo_;
  alignas(64) std::atomic<uint32_t> sequenceId{random<uint32_t>()};

  alignas(64) std::array<std::unique_ptr<RpcImplBase>, (int)ConnectionType::count> rpcs_;
  std::array<std::once_flag, (int)ConnectionType::count> rpcsInited_{};

  alignas(64) PeerId myId = PeerId::generate();
  std::string_view myName = persistentString(myId.toString());
  Function<void(const Error&)> onError_;

  alignas(64) SpinMutex listenersMutex_;
  std::array<Listener, (int)ConnectionType::count> listeners_;
  std::list<std::unique_ptr<Connection>> floatingConnections_;
  std::unordered_map<RpcConnectionImplBase*, decltype(floatingConnections_)::iterator> floatingConnectionsMap_;

  alignas(64) SpinMutex peersMutex_;
  std::unordered_map<std::string_view, PeerImpl> peers_;

  alignas(64) SpinMutex garbageMutex_;
  std::vector<std::unique_ptr<RpcConnectionImplBase>> garbageConnections_;
  std::vector<std::unique_ptr<RpcListenerImplBase>> garbageListeners_;
  SpinMutex findPeerMutex_;
  std::vector<std::string_view> findPeerList_;
  std::vector<std::string_view> findPeerLocalNameList_;
  std::vector<PeerImpl*> findPeerLocalPeerList_;

  std::atomic<std::chrono::steady_clock::time_point> lastRanMisc = std::chrono::steady_clock::time_point{};

  std::chrono::steady_clock::time_point lastPrint;

  std::atomic<bool> setupDone_ = false;
  std::atomic<bool> doingSetup_ = false;
  std::vector<ConnectionTypeInfo> info_;

  std::atomic_uint32_t connectionEnabledMask = 0;

  template<typename API>
  void tryInitRpc() {
    if (~connectionEnabledMask.load(std::memory_order_relaxed) & (1 << index<API>)) {
      return;
    }
    try {
      log("init %s\n", connectionTypeName[index<API>]);
      async::stopForksFromHereOn();
      auto u = std::make_unique<RpcImpl<API>>(*this);
      rpcs_[index<API>] = std::move(u);
    } catch (const std::exception& e) {
      log("Error during init of '%s': %s\n", connectionTypeName.at(index<API>), e.what());
    }
  }

  template<typename API>
  void lazyInitRpc() {
    std::call_once(rpcsInited_[index<API>], [this]() { tryInitRpc<API>(); });
  }

  template<typename F>
  struct FBuiltin : FBase {
    F f;
    template<typename F2>
    FBuiltin(F2&& f) : f(std::forward<F2>(f)) {}
    virtual ~FBuiltin() {}
    virtual void call(BufferHandle inbuffer, Function<void(BufferHandle)> callback) noexcept override {
      f(std::move(inbuffer), std::move(callback));
    }
  };

  template<typename F>
  void definebuiltin(std::string_view name, F&& f) {
    define(name, std::make_unique<FBuiltin<F>>(std::move(f)));
  }

  void setTransports(const std::vector<std::string>& names) {
    auto indexOf = [&](const std::string& name) {
      auto lowercase = [&](std::string str) {
        for (auto& v : str) {
          v = std::tolower(v);
        }
        return str;
      };
      for (size_t n = 0; n != 2; ++n) {
        const std::string& str = n == 0 ? name : lowercase(name);
        for (auto& list : {connectionShortTypeName, connectionTypeName}) {
          for (size_t i = 0; i != list.size(); ++i) {
            if (str == (n == 0 ? list[i] : lowercase(list[i]))) {
              return i;
            }
          }
        }
      }
      throw std::runtime_error("setTransports: transport '" + name + "' not found");
    };
    uint32_t mask = 0;
    for (auto& name : names) {
      mask |= 1 << indexOf(name);
    }
    connectionEnabledMask = mask;
  }

  Impl() {
    log("%p peer id is %s\n", (void*)this, myId.toString().c_str());
    incomingFifo_.next = &incomingFifo_;
    incomingFifo_.prev = &incomingFifo_;
    outgoingFifo_.next = &outgoingFifo_;
    outgoingFifo_.prev = &outgoingFifo_;

    uint32_t mask = 0;
    for (size_t i = 0; i != connectionDefaultEnabled.size(); ++i) {
      if (connectionDefaultEnabled[i]) {
        mask |= 1 << i;
      }
    }
    connectionEnabledMask = mask;

    definebuiltin("__reqFindFunction", [this](BufferHandle inbuffer, Function<void(BufferHandle)> callback) {
      // Look up function id by name
      uint32_t rid, fid;
      std::string_view name;
      deserializeBuffer(inbuffer, rid, fid, name);
      // log("find function '%s'\n", name);
      RemoteFunction rf;
      {
        std::lock_guard l(mutex_);
        auto i = funcIds_.find(name);
        if (i != funcIds_.end()) {
          rf.id = i->second;
        }
      }
      // log("returning fid %#x\n", rf.id);
      BufferHandle buffer;
      serializeToBuffer(buffer, (uint32_t)0, (uint32_t)Rpc::reqSuccess, rf);
      callback(std::move(buffer));
    });
  }
  virtual ~Impl() {
    terminate_ = true;
    if (timeoutThread_.joinable()) {
      timeoutSem_.post();
      timeoutThread_.join();
    }
    while (true) {
      // Calling close() on things shouldn't create any new connections as terminate_ is set,
      // but we loop just in case
      Deferrer defer;
      {
        std::lock_guard l2(garbageMutex_);
        std::lock_guard l(listenersMutex_);
        std::lock_guard l3(peersMutex_);
        for (auto& v : listeners_) {
          for (auto& v2 : v.listeners) {
            defer([ptr = &*v2] { ptr->close(); });
            garbageListeners_.push_back(std::move(v2));
          }
          v.listeners.clear();
        }
        for (auto& v : peers_) {
          for (auto& v2 : v.second.connections_) {
            std::lock_guard l4(v2.mutex);
            for (auto& v3 : v2.conns) {
              defer([ptr = &*v3] { ptr->close(); });
              garbageConnections_.push_back(std::move(v3));
            }
            v2.conns.clear();
          }
        }
        for (auto& v : floatingConnections_) {
          std::lock_guard l4(v->mutex);
          for (auto& v2 : v->conns) {
            defer([ptr = &*v2] { ptr->close(); });
            garbageConnections_.push_back(std::move(v2));
          }
        }
        floatingConnections_.clear();
      }
      if (defer.empty()) {
        break;
      }
    }

    {
      // collect will only defer calls to reconnect - we don't want that
      Deferrer defer(false);
      {
        std::unique_lock l(garbageMutex_);
        collect(l, garbageListeners_, defer);
        collect(l, garbageConnections_, defer);
      }
    }
    for (size_t i = 0; i != (size_t)ConnectionType::count; ++i) {
      rpcs_[i] = nullptr;
    }
  }

  template<typename T>
  auto& getBucket(T& arr, uint32_t rid) {
    return arr[(rid >> 1) % arr.size()];
  }

  template<typename T>
  std::chrono::steady_clock::time_point
  processTimeout(T& o, std::chrono::steady_clock::time_point now, Resend& s, Deferrer& defer) {
    log("Process timeout for rid %#x to %s\n", o.rid, o.peer->name);
    auto newTimeout = now + std::chrono::seconds(1);
    if (s.connection) {
      if (now - s.lastSendTimestamp <= std::chrono::milliseconds(250)) {
        return now + std::chrono::milliseconds(250);
      }
      auto timeout = std::chrono::milliseconds((int)std::ceil(
          o.peer->connections_.at(s.connectionIndex).runningLatency.load(std::memory_order_relaxed) * 2));
      if (timeout < std::chrono::seconds(1)) {
        timeout = std::chrono::seconds(1);
      }
      if (!s.hasAddedFailureLatency && now - s.lastSendTimestamp >= timeout) {
        s.hasAddedFailureLatency = true;
        log("  -- rid %#x to %s   %s failed \n", o.rid, o.peer->name, connectionTypeName.at(s.connectionIndex));
        switchOnAPI(
            (ConnectionType)s.connectionIndex, [&](auto api) { addLatency<decltype(api)>(*o.peer, now, timeout); });
      }
      if (now - s.connection->lastReceivedData.load(std::memory_order_relaxed) >= std::chrono::seconds(8)) {
        log("Closing connection %s to %s due to timeout!\n", connectionTypeName.at(s.connectionIndex), o.peer->name);
        auto& x = o.peer->connections_.at(s.connectionIndex);
        std::lock_guard l(x.mutex);
        for (size_t i = 0; i != x.conns.size(); ++i) {
          if (&*x.conns[i] == &*s.connection) {
            BufferHandle buffer;
            serializeToBuffer(buffer, (uint32_t)0, (uint32_t)Rpc::reqClose);
            switchOnAPI((ConnectionType)s.connectionIndex, [&](auto api) {
              ((RpcConnectionImpl<decltype(api)>&)*x.conns[i]).send(std::move(buffer), defer);
            });
            o.peer->throwAway(x, i);
            break;
          }
        }
        s.connection = nullptr;
      }
    }
    if (!s.connection) {
      s.pokeCount = 0;
      s.acked = false;
    }
    if (s.pokeCount < 2) {
      log("timeout sending poke for rid %#x (destined for %s)\n", o.rid, o.peer->name);
      BufferHandle buffer;
      serializeToBuffer(buffer, o.rid, Rpc::reqPoke);
      size_t index;
      bool b =
          o.peer->banditSend(connectionEnabledMask.load(std::memory_order_relaxed), std::move(buffer), defer, &index);
      // log("timeout bandit send result: %d\n", b);
      if (b) {
        if (s.pokeCount == 0) {
          s.pokeTimestamp = now;
        }
        ++s.pokeCount;
        ++s.totalPokeCount;

        newTimeout =
            now +
            std::chrono::milliseconds((int)std::ceil(
                o.peer->connections_.at(index).runningLatency.load(std::memory_order_relaxed) * (4 * o.timeoutCount)));
        newTimeout = std::max(newTimeout, now + std::chrono::milliseconds(s.acked ? 1000 : 100));
        newTimeout = std::min(newTimeout, now + std::chrono::seconds(2));
      } else {
        newTimeout = now + std::chrono::milliseconds(250);
      }
      if (s.totalPokeCount >= 4) {
        newTimeout = now + std::chrono::seconds(2);
      }
    }
    return newTimeout;
  }

  template<typename T>
  void processTimeout(std::chrono::steady_clock::time_point now, T& o, Deferrer& defer) {
    ++o.timeoutCount;
    auto newTimeout = now + std::chrono::seconds(1);
    // log("process timeout!\n");
    if (o.peer) {
      newTimeout = std::min(newTimeout, processTimeout(o, now, o.resend, defer));
    }
    o.timeout = newTimeout;
  }

  template<typename L, typename T>
  void collect(L& lock, T& ref, Deferrer& defer, bool respectTimeWait = false) noexcept {
    if (!ref.empty()) {
      auto now = std::chrono::steady_clock::now();
      T tmp;
      std::swap(ref, tmp);
      lock.unlock();
      for (auto& v : tmp) {
        if constexpr (std::is_same_v<std::remove_reference_t<decltype(v)>, std::unique_ptr<RpcConnectionImplBase>>) {

          auto checkBucket = [&](auto& bucket) {
            std::lock_guard l(bucket.mutex);
            for (auto& [id, x] : bucket.map) {
              if (x.resend.connection && &*x.resend.connection == &*v) {
                x.resend.connection = nullptr;
              }
            }
          };
          for (auto& bucket : incoming_) {
            checkBucket(bucket);
          }
          for (auto& bucket : outgoing_) {
            checkBucket(bucket);
          }

          if (respectTimeWait && now < v->timeWait) {
            // log("time wait %s connection %s <-> %s\n", connectionTypeName.at(v->apiIndex()), v->localAddr(),
            // v->remoteAddr());
            lock.lock();
            ref.push_back(std::move(v));
            lock.unlock();
          } else {
            log("collecting %s connection %s <-> %s\n", connectionTypeName.at(v->apiIndex()), v->localAddr(),
                v->remoteAddr());

            if (v->isExplicit) {
              switchOnAPI((ConnectionType)v->apiIndex(), [&](auto api) {
                log("Reconnecting to %s...\n", v->connectAddr);
                connect<decltype(api)>(v->connectAddr, defer);
              });
            }
          }
        }
      }
      tmp.clear();
      // log("GARBAGE COLLECTED YEY!\n");
      lock.lock();
    }
  }

  std::vector<std::unique_ptr<RpcConnectionImplBase>> toCollectTemporary;

  void collectFloatingConnections(std::chrono::steady_clock::time_point now) {
    std::unique_lock l(listenersMutex_);
    toCollectTemporary.clear();
    auto& toCollect = toCollectTemporary;
    for (auto i = floatingConnections_.begin(); i != floatingConnections_.end();) {
      auto& c = *i;
      if (now - c->creationTimestamp >= std::chrono::seconds(10)) {
        log("Collecting floating connection\n");
        for (auto& v2 : c->conns) {
          toCollect.push_back(std::move(v2));
        }
        for (auto i2 = floatingConnectionsMap_.begin(); i2 != floatingConnectionsMap_.end(); ++i2) {
          if (i2->second == i) {
            floatingConnectionsMap_.erase(i2);
            break;
          }
        }
        i = floatingConnections_.erase(i);
      } else {
        ++i;
      }
    }
    l.unlock();
    if (!toCollect.empty()) {
      std::lock_guard l(garbageMutex_);
      for (auto& v : toCollect) {
        garbageConnections_.push_back(std::move(v));
      }
    }
  }

  void collectGarbage() {
    Deferrer defer;
    {
      std::unique_lock l(garbageMutex_);
      collect(l, garbageConnections_, defer, true);
      collect(l, garbageListeners_, defer);
    }
  }

  void setTimeout(std::chrono::milliseconds milliseconds) {
    timeoutMilliseconds_.store(milliseconds.count(), std::memory_order_relaxed);
  }
  std::chrono::milliseconds getTimeout() {
    return std::chrono::milliseconds(timeoutMilliseconds_.load(std::memory_order_relaxed));
  }

  void debugInfo() {
    std::lock_guard l(peersMutex_);
    auto now = std::chrono::steady_clock::now();
    for (auto& v : peers_) {
      auto& p = v.second;
      std::lock_guard l(p.idMutex_);
      fmt::printf("Peer %s (%s)\n", std::string(p.name).c_str(), p.id.toString().c_str());
      for (size_t i = 0; i != p.connections_.size(); ++i) {
        auto& x = p.connections_[i];
        fmt::printf(
            " %s x%d  latency %g bandit %g\n", connectionTypeName[i], x.sendCount.load(std::memory_order_relaxed),
            x.runningLatency.load(), x.readBanditValue.load());
        fmt::printf("    %d conns:\n", x.conns.size());
        for (auto& v : x.conns) {
          float t = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(
                        now - v->lastReceivedData.load(std::memory_order_relaxed))
                        .count();
          fmt::printf(
              "      %s (%s, %s) [%s] age %g\n", v->dead ? "dead" : "alive", v->localAddr(), v->remoteAddr(),
              v->connectAddr, t);
        }
      }
    }
    fflush(stdout);
  }

  void timeoutThreadEntry() {
    async::setCurrentThreadName("timeout");
    while (!terminate_.load(std::memory_order_relaxed)) {
      Deferrer defer;
      auto now = std::chrono::steady_clock::now();
      if (lastRanMisc.load() + std::chrono::milliseconds(250) <= now) {
        lastRanMisc.store(now);
        collectFloatingConnections(now);
        collectGarbage();
        findPeersImpl(defer);
        defer.execute();
        now = std::chrono::steady_clock::now();
      }

      auto timeout = timeout_.load(std::memory_order_relaxed);
      timeout = std::min(timeout, now + std::chrono::seconds(1));
      while (now < timeout && !terminate_.load(std::memory_order_relaxed)) {
        timeoutSem_.wait_for(timeout - now);
        now = std::chrono::steady_clock::now();
        continue;
      }
      auto newTimeout = now + std::chrono::seconds(5);
      timeout_.store(newTimeout);
      auto process = [&](auto& container) {
        auto absTimeoutDuration = std::chrono::milliseconds(timeoutMilliseconds_.load(std::memory_order_relaxed));
        for (auto& b : container) {
          std::unique_lock l(b.mutex);
          bool anyToRemove = false;
          for (auto& v : b.map) {
            if (now - v.second.creationTimestamp >= absTimeoutDuration) {
              anyToRemove = true;
            }
            if (v.second.resend.buffer) {
              if (now >= v.second.timeout) {
                processTimeout(now, v.second, defer);
              }
              newTimeout = std::min(newTimeout, v.second.timeout);
            }

            //            constexpr bool isIncoming = std::is_same_v<std::decay_t<decltype(v.second)>, Incoming>;
            //            float t = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(now -
            //            v.second.creationTimestamp).count(); if constexpr (isIncoming) {
            //              log("Response %#x age: %g\n", v.second.rid, t);
            //            } else {
            //              log("Request %#x age: %g\n", v.second.rid, t);
            //            }
          }
          if (anyToRemove) {
            std::unique_lock l2(incomingFifoMutex_, std::defer_lock);
            constexpr bool isIncoming = std::is_same_v<std::decay_t<decltype(b.map.begin()->second)>, Incoming>;
            if (isIncoming) {
              if (!l2.try_lock()) {
                l.unlock();
                l2.lock();
                l.lock();
              }
            }
            for (auto i = b.map.begin(); i != b.map.end();) {
              auto& v = i->second;
              if (now - v.creationTimestamp >= absTimeoutDuration) {
                if constexpr (isIncoming) {
                  if (v.resend.buffer) {
                    listErase(&v);
                  }
                  log("Response %#x timed out for real\n", v.rid);

                  v.peer->addRecentIncoming(v.rid, now + std::chrono::minutes(1));
                } else {
                  log("Request %#x timed out for real\n", v.rid);
                  Error err(fmt::sprintf("Call (%s::%s) timed out", v.peer->name, v.peer->functionName(v.fid)));
                  std::move(v.response)(nullptr, &err);
                }
                if (v.resend.connection) {
                  log("sent over %s\n", connectionTypeName.at(v.resend.connectionIndex));
                } else {
                  log("null connection\n");
                }
                cleanup(v, defer);
                i = b.map.erase(i);
              } else {
                ++i;
              }
            }
          }
        }
      };
      process(outgoing_);
      process(incoming_);
      defer.execute();
      timeout = timeout_.load(std::memory_order_relaxed);
      while (newTimeout < timeout && !timeout_.compare_exchange_weak(timeout, newTimeout))
        ;
      // log("new timeout is in %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(newTimeout - now).count());

      //      if (now - lastPrint >= std::chrono::seconds(30)) {
      //        lastPrint = now;
      //        std::lock_guard l(peersMutex_);
      //        for (auto& v : peers_) {
      //          auto& p = v.second;
      //          std::lock_guard l(p.idMutex_);
      //          log("Peer %s (%s)\n", std::string(p.name).c_str(), p.id.toString().c_str());
      //          for (size_t i = 0; i != p.connections_.size(); ++i) {
      //            auto& x = p.connections_[i];
      //            log(" %s x%d  latency %g bandit %g\n", connectionTypeName[i],
      //            x.sendCount.load(std::memory_order_relaxed), x.runningLatency.load(), x.readBanditValue.load());
      //          }
      //        }
      //      }
    }
  }

  void startTimeoutThread() {
    timeoutThread_ = std::thread([this]() noexcept { timeoutThreadEntry(); });
  }

  uint32_t getFunctionId(std::string_view name) {
    return baseFuncId_ + std::hash<std::string_view>()(name) % maxFunctions_;
  }

  void define(std::string_view name, std::unique_ptr<Rpc::FBase>&& f) {
    name = persistentString(name);
    std::lock_guard l(mutex_);
    uint32_t id = getFunctionId(name);
    // log("Define function %s with id %#x\n", name, id);
    if (funcIds_.find(name) != funcIds_.end()) {
      throw Error("Function " + std::string(name) + " already defined");
    }
    if (funcs_.find(id) != funcs_.end()) {
      throw Error("Function name " + std::string(name) + " has a hash collision with another function name");
    }
    funcIds_[name] = id;
    funcs_[id] = std::move(f);
  }
  std::string_view persistentString(std::string_view name) {
    std::lock_guard l(mutex_);
    auto i = stringMap_.find(name);
    if (i != stringMap_.end()) {
      return *i;
    }
    stringList_.emplace_back(name);
    return *stringMap_.emplace(stringList_.back()).first;
  }

  template<typename API>
  void setup() noexcept {
    lazyInitRpc<API>();
    auto& x = listeners_.at(index<API>);
    if (x.implicitCount > 0) {
      return;
    }
    for (auto& addr : API::defaultAddr()) {
      listen<API, false>(addr);
    }
  }

  auto& getInfo() noexcept {
    if (!setupDone_) {
      if (doingSetup_.exchange(true)) {
        while (!setupDone_) {
          std::this_thread::yield();
        }
      } else {
        for (size_t i = 0; i != (size_t)ConnectionType::count; ++i) {
          switchOnAPI(ConnectionType(i), [&](auto api) { setup<decltype(api)>(); });
        }

        std::lock_guard l(listenersMutex_);
        info_.clear();
        for (size_t i = 0; i != listeners_.size(); ++i) {
          ConnectionTypeInfo ci;
          ci.name = connectionTypeName.at(i);
          for (auto& v : listeners_[i].listeners) {
            try {
              const auto& str = v->localAddr();
              if (!str.empty()) {
                ci.addr.push_back(persistentString(str));
              }
            } catch (const std::exception& e) {
            }
          }
          info_.push_back(std::move(ci));
        }

        setupDone_ = true;
      }
    }
    return info_;
  }

  template<typename API, bool explicit_ = true>
  void connect(std::string_view addr, Deferrer& defer) {
    defer([addr = std::string(addr), this] {
      if (terminate_.load(std::memory_order_relaxed)) {
        return;
      }
      Deferrer defer;
      {
        log("Connecting with %s to %s\n", connectionTypeName[index<API>], addr);
        lazyInitRpc<API>();
        auto* u = getImpl<API>();
        if (!u) {
          if constexpr (explicit_) {
            throw std::runtime_error("Backend " + std::string(connectionTypeName.at(index<API>)) + " is not available");
          } else {
            fatal("Backend %s is not available", connectionTypeName.at(index<API>));
          }
        }

        getInfo();

        auto connection = u->context.connect(std::string(addr));

        auto c = std::make_unique<Connection>();
        std::unique_lock l(listenersMutex_);
        if (terminate_.load(std::memory_order_relaxed)) {
          return;
        }
        c->outgoing = true;
        c->isExplicit = explicit_;
        c->addr = persistentString(addr);
        // RpcConnectionImpl<API> xx(*u, u->context.connect(std::string(addr)));
        auto cu = std::make_unique<RpcConnectionImpl<API>>(*u, std::move(connection));
        cu->isExplicit = explicit_;
        cu->connectAddr = addr;
        cu->greet(myName, myId, info_, defer);
        cu->start(defer);
        c->conns.push_back(std::move(cu));
        floatingConnections_.push_back(std::move(c));
        floatingConnectionsMap_[&*floatingConnections_.back()->conns.back()] = std::prev(floatingConnections_.end());
      }
    });
  }

  template<typename API, bool explicit_ = true>
  auto listen(std::string_view addr) {
    if constexpr (explicit_) {
      if (terminate_.load(std::memory_order_relaxed)) {
        return;
      }
    }
    lazyInitRpc<API>();
    auto* u = getImpl<API>();
    if (!u) {
      if constexpr (!explicit_) {
        return false;
      } else {
        fatal("Backend is not available");
      }
    }
    decltype(u->context.listen({std::string(addr)})) ul;
    try {
      ul = u->context.listen({std::string(addr)});
    } catch (const std::exception& e) {
      log("error in listen<%s, %d>(%s): %s\n", connectionTypeName[index<API>], explicit_, addr, e.what());
      if constexpr (!explicit_) {
        return false;
      } else {
        throw;
      }
    }
    std::lock_guard l(listenersMutex_);
    auto& x = listeners_.at(index<API>);
    std::unique_ptr<RpcListenerImpl<API>> i;
    try {
      i = std::make_unique<RpcListenerImpl<API>>(*u, std::move(ul), addr);
      log("%s::listen<%s, %d>(%s) success\n", myName, connectionTypeName[index<API>], explicit_, addr);
    } catch (const std::exception& e) {
      std::lock_guard l(garbageMutex_);
      if (i) {
        garbageListeners_.push_back(std::move(i));
      }
      log("error in listen<%s, %d>(%s): %s\n", connectionTypeName[index<API>], explicit_, addr, e.what());
      if constexpr (!explicit_) {
        return false;
      } else {
        throw;
      }
    }
    i->active = true;
    i->isExplicit = explicit_;
    ++x.activeCount;
    ++(explicit_ ? x.explicitCount : x.implicitCount);
    x.listeners.push_back(std::move(i));
    if constexpr (!explicit_) {
      return true;
    }
  }

  void setOnError(Function<void(const Error&)>&& callback) {
    if (onError_) {
      throw std::runtime_error("onError callback already set");
    }
    onError_ = std::move(callback);
  }

  template<typename API>
  auto* getImpl() {
    return (RpcImpl<API>*)&*rpcs_.at(index<API>);
  }

  bool resend(PeerImpl& peer, Resend& s, Deferrer& defer) {
    size_t index;
    Me<RpcConnectionImplBase> connection;
    if (peer.banditSend(connectionEnabledMask.load(std::memory_order_relaxed), s.buffer, defer, &index, &connection)) {
      // log("resend %#x %#x success\n", rid, fid);
      s.lastSendTimestamp = std::chrono::steady_clock::now();
      s.connection = std::move(connection);
      s.connectionIndex = index;
      return true;
    } else {
      // log("resend %#x %#x failure\n", rid, fid);
      s.lastSendFailTimestamp = std::chrono::steady_clock::now();
      s.connection = nullptr;
      return false;
    }
  }

  void resetSequenceId() {
    sequenceId += random<uint32_t>(0, 0x100) * 0x100;
  }

  void sendRequest(PeerImpl& peer, uint32_t fid, BufferHandle buffer, rpc::Rpc::ResponseCallback response) noexcept {
    if (buffer->size != (uint32_t)buffer->size) {
      fatal("RPC request is too large!");
    }
    auto* ptr = dataptr<std::byte>(&*buffer);
    uint32_t baseRid = sequenceId.fetch_add(1, std::memory_order_relaxed);
    uint32_t rid = baseRid << 1 | 1;
    if (baseRid % 0x100 == 0xff) {
      resetSequenceId();
    }
    auto* ridPtr = ptr;
    std::memcpy(ptr, &rid, sizeof(rid));
    ptr += sizeof(rid);
    std::memcpy(ptr, &fid, sizeof(fid));
    ptr += sizeof(fid);

    auto now = std::chrono::steady_clock::now();
    Deferrer defer;
    {
      log("send request %#x %s::%#x\n", rid, peer.name, fid);
      SharedBufferHandle shared(buffer.release());
      auto& oBucket = getBucket(outgoing_, rid);
      std::unique_lock l(oBucket.mutex);
      auto in = oBucket.map.try_emplace(rid);
      while (!in.second) {
        baseRid = sequenceId.fetch_add(1, std::memory_order_relaxed);
        rid = baseRid << 1 | 1;
        if (baseRid % 0x100 == 0xff) {
          resetSequenceId();
        }
        std::memcpy(ridPtr, &rid, sizeof(rid));
        in = oBucket.map.try_emplace(rid);
      }
      // log("sending request with rid %#x\n", rid);
      auto& q = in.first->second;
      q.rid = rid;
      q.fid = fid;
      q.peer = &peer;
      q.requestTimestamp = now;
      q.timeout = now + std::chrono::milliseconds(100);
      q.response = std::move(response);
      q.resend.buffer = shared;
      resend(peer, q.resend, defer);
    }
    defer.execute();
    updateTimeout(now + std::chrono::seconds(1));
  }

  void updateTimeout(std::chrono::steady_clock::time_point myTimeout) {
    static_assert(std::atomic<std::chrono::steady_clock::time_point>::is_always_lock_free);
    auto timeout = timeout_.load(std::memory_order_acquire);
    while (myTimeout < timeout) {
      if (timeout_.compare_exchange_weak(timeout, myTimeout)) {
        timeoutSem_.post();
        break;
      }
    }
    std::call_once(timeoutThreadOnce_, [&]() { startTimeoutThread(); });
  }

  PeerImpl& getPeer(std::string_view name) {
    std::lock_guard l(peersMutex_);
    auto i = peers_.try_emplace(name, *this);
    auto& p = i.first->second;
    if (i.second) {
      p.name = persistentString(name);
      const_cast<std::string_view&>(i.first->first) = p.name;
    }
    return p;
  }

  void sendRequest(
      std::string_view peerName, std::string_view functionName, BufferHandle buffer,
      ResponseCallback response) noexcept {
    log("sendRequest %s::%s %d bytes\n", peerName, functionName, buffer->size);
    if (peerName == myName) {
      fatal("Attempt to call function %s on myself! Self-calling is not supported", functionName);
    }
    auto& peer = getPeer(peerName);
    uint32_t fid = peer.functionId(functionName);
    if (fid == 0) {
      functionName = persistentString(functionName);
      BufferHandle buffer2;
      serializeToBuffer(buffer2, (uint32_t)0, (uint32_t)0, functionName);
      sendRequest(
          peer, getFunctionId("__reqFindFunction"), std::move(buffer2),
          [this, peer = &peer, functionName = functionName, buffer = std::move(buffer),
           response = std::move(response)](BufferHandle recvbuffer, Error* error) mutable noexcept {
            if (error) {
              std::move(response)(nullptr, error);
            } else {
              // log("got %d bytes\n", recvbuffer->size);
              uint32_t xrid, xfid;
              RemoteFunction rf;
              deserializeBuffer(recvbuffer, xrid, xfid, rf);
              uint32_t fid = rf.id;
              // log("got id %#x\n", id);
              if (fid == 0) {
                Error err(
                    "RPC remote function " + std::string(peer->name) + "::'" + std::string(functionName) +
                    "' does not exist");
                std::move(response)(nullptr, &err);
                return;
              }
              rf.typeId = persistentString(rf.typeId);
              peer->setRemoteFunc(functionName, rf);
              sendRequest(*peer, fid, std::move(buffer), std::move(response));
            }
          });
    } else {
      sendRequest(peer, fid, std::move(buffer), std::move(response));
    }
  }

  template<typename API>
  void onAccept(RpcListenerImpl<API>& listener, std::unique_ptr<RpcConnectionImpl<API>> conn, Error* err) {
    log("onAccept!()\n");
    Deferrer defer;
    {
      getInfo();
      std::unique_lock l(listenersMutex_);
      if (terminate_.load(std::memory_order_relaxed)) {
        return;
      }
      if (err) {
        log("accept error: %s\n", err->what());
        listener.active = false;
        bool isExplicit = listener.isExplicit;
        auto& x = listeners_.at(index<API>);
        --x.activeCount;
        --(isExplicit ? x.explicitCount : x.implicitCount);
        if (isExplicit) {
          if (onError_) {
            onError_(*err);
          }
          //        int nExplicit = 0;
          //        for (auto& v : listeners_) {
          //          if (v.explicitCount) {
          //            ++nExplicit;
          //          }
          //        }
          //        if (nExplicit == 0) {
          //          l.unlock();
          //          if (onError_) {
          //            onError_(*err);
          //          }
          //        }
        }
      } else {
        log("accept got connection!\n");
        auto c = std::make_unique<Connection>();
        conn->greet(myName, myId, info_, defer);
        conn->start(defer);
        c->conns.push_back(std::move(conn));
        floatingConnections_.push_back(std::move(c));
        floatingConnectionsMap_[&*floatingConnections_.back()->conns.back()] = std::prev(floatingConnections_.end());
      }
    }
  }

  void setName(std::string_view name) {
    myName = persistentString(name);
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

  template<typename API>
  void onGreeting(
      RpcConnectionImpl<API>& conn, std::string_view peerName, PeerId peerId, std::vector<ConnectionTypeInfo>&& info) {
    log("%s::%s::onGreeting!(\"%s\", %s)\n", std::string(myName).c_str(), connectionTypeName[index<API>],
        std::string(peerName).c_str(), peerId.toString().c_str());
    for (auto& v : info) {
      log(" %s\n", std::string(v.name).c_str());
      for (auto& v2 : v.addr) {
        log("  @ %s\n", std::string(v2).c_str());
      }
    }
    Deferrer defer;
    {
      std::unique_lock l(listenersMutex_);
      if (terminate_.load(std::memory_order_relaxed)) {
        return;
      }
      auto i = floatingConnectionsMap_.find(&conn);
      if (i != floatingConnectionsMap_.end()) {
        auto i2 = i->second;
        floatingConnectionsMap_.erase(i);
        auto cptr = std::move(*i2);
        floatingConnections_.erase(i2);

        l.unlock();

        if (peerId == myId) {
          std::lock_guard l(garbageMutex_);
          for (auto& c : cptr->conns) {
            garbageConnections_.push_back(std::move(c));
          }
          log("I connected to myself! oops!\n");
          return;
        }
        if (peerName == myName) {
          std::lock_guard l(garbageMutex_);
          for (auto& c : cptr->conns) {
            garbageConnections_.push_back(std::move(c));
          }
          log("Peer with same name as me! Refusing connection!\n");
          return;
        }

        PeerImpl& peer = getPeer(peerName);
        {
          std::lock_guard l(peer.idMutex_);
          peer.id = peerId;
          peer.hasId = true;
          peer.findThisPeerIncrementingTimeoutMilliseconds.store(250, std::memory_order_relaxed);
        }
        if (&conn != &*cptr->conns.back()) {
          fatal("onGreeting internal error conns mismatch");
        }
        conn.peer = &peer;
        std::unique_ptr<RpcConnectionImplBase> oldconn;
        {
          auto& x = peer.connections_[index<API>];
          std::lock_guard l(x.mutex);
          x.isExplicit = cptr->isExplicit;
          x.outgoing = cptr->outgoing;
          x.addr = std::move(cptr->addr);
          for (auto& c : cptr->conns) {
            x.conns.push_back(std::move(c));
          }
          x.hasConn = true;
          x.valid = true;
        }
        if (oldconn) {
          std::lock_guard l(garbageMutex_);
          garbageConnections_.push_back(std::move(oldconn));
        }

        {
          std::lock_guard l(peer.idMutex_);
          for (auto& v : info) {
            for (size_t i = 0; i != connectionTypeName.size(); ++i) {
              if (v.name == connectionTypeName[i]) {
                auto& x = peer.connections_.at(i);
                auto trimAddresses = [&](int n) {
                  if (x.remoteAddresses.size() > n) {
                    x.remoteAddresses.erase(
                        x.remoteAddresses.begin(), x.remoteAddresses.begin() + (x.remoteAddresses.size() - 24));
                  }
                };
                std::lock_guard l(x.mutex);
                x.valid = true;
                std::string addr;
                if (API::addressIsIp && addressIsIp((ConnectionType)i)) {
                  addr = conn.remoteAddr();
                }
                if (!addr.empty()) {
                  auto remote = decodeIpAddress(addr);
                  bool remoteIpv6 = remote.first.find(':') != std::string_view::npos;
                  for (auto& v2 : v.addr) {
                    auto v3 = decodeIpAddress(v2);
                    bool ipv6 = v3.first.find(':') != std::string_view::npos;
                    if (ipv6 != remoteIpv6) {
                      continue;
                    }
                    std::string newAddr = (ipv6 ? "[" + std::string(remote.first) + "]" : std::string(remote.first)) +
                                          ":" + std::to_string(v3.second);
                    if (std::find(x.remoteAddresses.begin(), x.remoteAddresses.end(), newAddr) ==
                        x.remoteAddresses.end()) {
                      x.remoteAddresses.push_back(persistentString(newAddr));
                      trimAddresses(48);
                    }
                  }
                } else if (!addressIsIp((ConnectionType)i)) {
                  for (auto& v2 : v.addr) {
                    if (std::find(x.remoteAddresses.begin(), x.remoteAddresses.end(), v2) == x.remoteAddresses.end()) {
                      x.remoteAddresses.push_back(persistentString(v2));
                      trimAddresses(48);
                    }
                  }
                }
                for (auto& v : x.remoteAddresses) {
                  log(" -- %s -- has a remote address %s\n", std::string(peer.name).c_str(), std::string(v).c_str());
                }
                trimAddresses(24);
              }
            }
          }
        }

        for (auto& b : outgoing_) {
          std::lock_guard l(b.mutex);
          for (auto& v : b.map) {
            auto& o = v.second;
            if (o.peer == &peer && !o.resend.connection) {
              // log("poking on newly established connection\n");
              BufferHandle buffer;
              serializeToBuffer(buffer, o.rid, Rpc::reqPoke);
              conn.send(std::move(buffer), defer);
            }
          }
        }
      }
    }
  }

  void findPeersImpl(Deferrer& defer) {
    std::vector<std::string_view> nameList;
    std::vector<PeerImpl*> peerList;
    {
      std::lock_guard l(findPeerMutex_);
      if (findPeerList_.empty()) {
        return;
      }
      std::swap(nameList, findPeerLocalNameList_);
      std::swap(nameList, findPeerList_);
      std::swap(peerList, findPeerLocalPeerList_);
      findPeerList_.clear();
      peerList.clear();
    }
    size_t n = 0;
    {
      std::lock_guard l(peersMutex_);
      n = peers_.size();
    }
    peerList.reserve(n + n / 4);
    {
      std::lock_guard l(peersMutex_);
      for (auto& v : peers_) {
        peerList.push_back(&v.second);
      }
    }
    auto now = std::chrono::steady_clock::now();
    peerList.erase(
        std::remove_if(
            peerList.begin(), peerList.end(),
            [&](PeerImpl* p) {
              if (!p->hasId.load(std::memory_order_relaxed)) {
                return true;
              }
              for (auto& v : p->connections_) {
                if (p->isConnected(v)) {
                  return false;
                }
              }
              if (now - p->lastFindPeers <= std::chrono::seconds(2)) {
                return false;
              }
              return true;
            }),
        peerList.end());

    log("findPeers has %d/%d peers with live connection\n", peerList.size(), n);

    bool anySuccess = false;

    if (!peerList.empty()) {
      size_t nToKeep = std::min((size_t)std::ceil(std::log2(n)), peerList.size());
      nToKeep = std::max(nToKeep, std::min(n, (size_t)2));
      while (peerList.size() > nToKeep) {
        std::swap(peerList.back(), peerList.at(random<size_t>(0, peerList.size() - 1)));
        peerList.pop_back();
      }
      log("looking among %d peers\n", peerList.size());
      BufferHandle buffer;
      serializeToBuffer(buffer, (uint32_t)1, (uint32_t)Rpc::reqLookingForPeer, nameList);
      SharedBufferHandle shared{buffer.release()};
      for (auto* p : peerList) {
        p->lastFindPeers = now;
        anySuccess |= p->banditSend(
            connectionEnabledMask.load(std::memory_order_relaxed), shared, defer, nullptr, nullptr, false);
      }
    }

    if (anySuccess) {
      std::lock_guard l(findPeerMutex_);
      std::swap(nameList, findPeerLocalNameList_);
      std::swap(peerList, findPeerLocalPeerList_);
    } else {
      // log("No connectivity to any peers for search; keeping find list\n");
      std::lock_guard l(findPeerMutex_);
      std::swap(nameList, findPeerList_);
      std::swap(peerList, findPeerLocalPeerList_);
    }
  }

  void findPeer(std::string_view name) {
    if (name == myName) {
      return;
    }
    std::call_once(timeoutThreadOnce_, [&]() { startTimeoutThread(); });
    std::lock_guard l(findPeerMutex_);
    if (std::find(findPeerList_.begin(), findPeerList_.end(), name) == findPeerList_.end()) {
      log("looking for %s\n", std::string(name).c_str());
      findPeerList_.push_back(name);
    }
    timeoutSem_.post();
  }

  template<typename API>
  void
  addLatency(PeerImpl& peer, std::chrono::steady_clock::time_point now, std::chrono::steady_clock::duration duration) {
    uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();

    float latency = std::min(us / 1000.0f, 10000.0f);

    Connection& cx = peer.connections_[index<API>];
    std::lock_guard l(cx.latencyMutex);

    float t =
        std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(now - cx.lastUpdateLatency).count();
    cx.lastUpdateLatency = now;
    float a = std::pow(0.9375f, std::min(t, 2.0f));

    float runningLatency = cx.runningLatency.load(std::memory_order_relaxed);
    runningLatency = runningLatency * a + latency * (1.0f - a);
    cx.runningLatency.store(runningLatency, std::memory_order_relaxed);
    float min = runningLatency;
    for (auto& v : peer.connections_) {
      if (v.valid.load(std::memory_order_relaxed) && v.hasConn.load(std::memory_order_relaxed)) {
        float l = v.runningLatency.load(std::memory_order_relaxed);
        // log("running latency for %d is %g\n", &v - peer.connections_.data(), l);
        min = std::min(min, l);
      }
    }
    // log("runningLatency is %g\n", runningLatency);
    float r = runningLatency <= min ? 1.0f : -1.0f;
    float banditValue = cx.writeBanditValue * a + r * (1.0f - a);
    cx.writeBanditValue = banditValue;
    if (std::abs(banditValue - cx.readBanditValue.load(std::memory_order_relaxed)) >= 0.001f) {
      // log("update bandit value %g -> %g\n", cx.readBanditValue.load(), banditValue);
      cx.readBanditValue.store(banditValue, std::memory_order_relaxed);
    } else {
      // log("bandit value of %g does not need update\n", cx.readBanditValue.load(std::memory_order_relaxed));
    }
  }

  template<typename... Args>
  void log(const char* fmt, Args&&... args) {
    auto s = fmt::sprintf(fmt, std::forward<Args>(args)...);
    rpc::log("%s: %s", myName, s);
  }

  void cleanup(Rpc::Impl::Incoming& o, Deferrer& defer) {
    defer([resend = std::move(o.resend.buffer), recv = std::move(o.recv.buffer)] {});
  }

  void cleanup(Rpc::Impl::Outgoing& o, Deferrer& defer) {
    defer([resend = std::move(o.resend.buffer), recv = std::move(o.recv.buffer), response = std::move(o.response)] {});
  }
};

template<typename API>
struct RpcImpl : RpcImplBase {
  typename API::Context context;

  RpcImpl(Rpc::Impl& rpc) : RpcImplBase(rpc) {}

  void onAccept(RpcListenerImpl<API>& listener, std::unique_ptr<RpcConnectionImpl<API>>&& conn, Error* err) {
    rpc.onAccept(listener, std::move(conn), err);
  }

  void onGreeting(
      RpcConnectionImpl<API>& conn, std::string_view peerName, PeerId peerId, std::vector<ConnectionTypeInfo>&& info) {
    rpc.onGreeting(conn, peerName, peerId, std::move(info));
  }

  template<typename T>
  void handlePoke(T& container, PeerImpl& peer, RpcConnectionImpl<API>& conn, uint32_t rid, Deferrer& defer) {
    auto& bucket = rpc.getBucket(container, rid);
    std::unique_lock l(bucket.mutex);
    auto i = bucket.map.find(rid);
    if (i == bucket.map.end()) {
      // log("got poke for unknown rid, nack %d\n", partIndex);
      BufferHandle buffer;
      serializeToBuffer(buffer, rid, Rpc::reqNack);
      conn.send(std::move(buffer), defer);
    } else {
      auto& x = i->second;
      if (x.rid != rid) {
        fatal("handlePoke internal error: rid %#x is not set!\n", rid);
      }
      if (x.peer != &peer) {
        log("peer %p vs %p\n", (void*)x.peer, (void*)&peer);
        log("rid collision on poke! (not fatal!)\n");
        return;
      }
      bool ack = false;
      if (x.recv.done) {
        ack = true;
      } else {
        ack = x.recv.buffer != nullptr;
        l.unlock();
      }
      // log("got poke %s rid %#x %d\n", ack ? "ack" : "nack", rid, partIndex);
      BufferHandle buffer;
      serializeToBuffer(buffer, rid, ack ? Rpc::reqAck : Rpc::reqNack);
      conn.send(std::move(buffer), defer);
    }
  }

  template<bool allowNew, typename T>
  bool handleRecv(T& container, PeerImpl& peer, RpcConnectionImpl<API>& conn, uint32_t rid, Deferrer& defer) {
    rpc.log("handleRecv peer %s rid %#x\n", peer.name, rid);
    auto find = [&](auto& bucket, auto& l) {
      auto check = [&](auto i) -> decltype(&i->second) {
        if (i == bucket.map.end()) {
          // ack if this is a response to an unknown request.
          // Otherwise, the remote peer will just keep sending it.
          l.unlock();
          BufferHandle buffer;
          serializeToBuffer(buffer, rid, Rpc::reqAck);
          conn.send(std::move(buffer), defer);
          return nullptr;
        }
        auto& x = i->second;
        if (x.rid != rid) {
          fatal("handleRecv internal error: rid %#x is not set!\n", rid);
        }
        if (x.peer != &peer) {
          log("peer %p vs %p\n", (void*)x.peer, (void*)&peer);
          log("rid collision on recv! (not fatal!)\n");
          return nullptr;
        }
        if (x.recv.done) {
          l.unlock();
          log("recv for rid %#x already done\n", rid);
          BufferHandle buffer;
          serializeToBuffer(buffer, rid, Rpc::reqAck);
          conn.send(std::move(buffer), defer);
          return nullptr;
        }
        return &x;
      };
      if constexpr (allowNew) {
        auto i = bucket.map.find(rid);
        if (i == bucket.map.end()) {
          std::lock_guard ril(peer.recentIncomingMutex);
          if (!peer.recentIncomingList.empty()) {
            peer.clearRecentIncomingTimeouts();
            if (!peer.recentIncomingList.empty()) {
              if (peer.recentIncomingMap.find(rid) != peer.recentIncomingMap.end()) {
                rpc.log("rid %#x recently handled; ignoring\n", rid);
                return (Rpc::Impl::Incoming*)nullptr;
              }
            }
          }
          auto i = bucket.map.try_emplace(rid);
          auto& x = i.first->second;
          if (i.second) {
            rpc.log("NEW rid %#x\n", rid);
            x.peer = &peer;
            x.rid = rid;
          } else {
            fatal("bucket map not found but created!?");
          }
          return check(i.first);
        } else {
          return check(i);
        }
      } else {
        return check(bucket.map.find(rid));
      }
    };

    // log("recv part 0 of rid %#x\n", rid);
    auto& bucket = rpc.getBucket(container, rid);
    std::unique_lock l(bucket.mutex);
    auto xptr = find(bucket, l);
    if (xptr) {
      // log("recv %d tensors\n", nTensors);
      auto& x = *xptr;
      x.recv.done = true;
      l.unlock();
      BufferHandle buffer;
      serializeToBuffer(buffer, rid, Rpc::reqAck);
      conn.send(std::move(buffer), defer);
      return true;
    }
    return false;
  }

  template<bool isIncoming, typename T>
  void handleAck(T& container, PeerImpl& peer, uint32_t rid, Deferrer& defer) {
    auto& bucket = rpc.getBucket(container, rid);
    std::optional<std::chrono::steady_clock::duration> duration;
    {
      std::unique_lock l2(rpc.incomingFifoMutex_, std::defer_lock);
      if constexpr (isIncoming) {
        l2.lock();
      }
      std::lock_guard l(bucket.mutex);
      auto i = bucket.map.find(rid);
      if (i != bucket.map.end()) {
        auto& x = i->second;

        auto now = std::chrono::steady_clock::now();
        auto& s = x.resend;
        if (!s.acked) {
          log("handleAck got ack for peer %s rid %#x\n", peer.name, rid);
          s.nackCount = 0;
          s.acked = true;
          s.ackTimestamp = now;
          duration = now - s.lastSendTimestamp;
        }

        if constexpr (isIncoming) {
          if (x.resend.buffer) {
            rpc.totalResponseSize_.fetch_sub(x.resend.buffer->size, std::memory_order_relaxed);
          }
          peer.addRecentIncoming(rid, now + std::chrono::minutes(1));
          listErase(&x);
          log("peer %s rid %#x acked and freed\n", peer.name, rid);
          rpc.cleanup(x, defer);
          bucket.map.erase(i);
        }
      }
    }
    if (duration) {
      rpc.addLatency<API>(peer, std::chrono::steady_clock::now(), *duration);
    }
  }

  template<typename T>
  void handleNack(T& container, PeerImpl& peer, uint32_t rid, Deferrer& defer) {
    log("got nack peer %s rid %#x\n", peer.name, rid);
    auto& bucket = rpc.getBucket(container, rid);
    std::unique_lock l(bucket.mutex);
    auto i = bucket.map.find(rid);
    if (i != bucket.map.end() && i->second.peer != &peer) {
      log("rid collision on nack! (not fatal error)\n");
      return;
    }
    if (i != bucket.map.end() && i->second.peer == &peer) {
      auto& x = i->second;
      auto& s = x.resend;
      if (s.buffer) {
        ++s.nackCount;
        // log("nackCount is now %d\n", s.nackCount);
        if (!s.connection) {
          // log("nack %#x %d resend\n", rid, partIndex);
          rpc.resend(peer, s, defer);
        } else {
          // log("nack %#x %d but resend already in progress\n", rid, partIndex);
        }
      }
    }
  }

  void onRequest(
      PeerImpl& peer, RpcConnectionImpl<API>& conn, uint32_t rid, uint32_t fid, const std::byte* ptr, size_t len,
      BufferHandle buffer, Deferrer& defer) {
    if (rpc.terminate_.load(std::memory_order_relaxed)) {
      return;
    }
    log("onRequest peer %s rid %#x (%#x) fid %#x\n", peer.name, rid, rid & ~(uint32_t)1, fid);
    rid &= ~(uint32_t)1;
    switch (fid) {
    case Rpc::reqAck: {
      // Peer acknowledged that it has received the response
      // (return value of an RPC call)
      handleAck<true>(rpc.incoming_, peer, rid, defer);
      break;
    }
    case Rpc::reqPoke: {
      // Peer is poking us to check the status of an RPC call
      // log("got poke for %#x\n", rid);
      handlePoke(rpc.incoming_, peer, conn, rid, defer);
      break;
    }
    case Rpc::reqNack: {
      // Peer nacked a poke from us; this means we may need to
      // resend a rpc response
      handleNack(rpc.incoming_, peer, rid, defer);
      break;
    }
    case Rpc::reqLookingForPeer: {
      // Peer is looking for some other peer(s)
      std::vector<std::string_view> names;
      deserializeBuffer(ptr, len, names);
      std::vector<PeerImpl*> foundPeers;
      std::unordered_map<std::string_view, std::vector<ConnectionTypeInfo>> info;
      {
        std::lock_guard l(rpc.peersMutex_);
        for (auto name : names) {
          auto i = rpc.peers_.find(name);
          if (i != rpc.peers_.end() && i->second.hasId.load(std::memory_order_relaxed)) {
            log("Peer '%s' is looking for '%s', and we know them!\n", peer.name, name);
            foundPeers.push_back(&i->second);
          } else {
            log("Peer '%s' is looking for '%s', but we don't know them :/\n", peer.name, name);
          }
        }
      }
      if (!foundPeers.empty()) {
        for (auto* ptr : foundPeers) {
          PeerImpl& p = *ptr;
          std::lock_guard l(p.idMutex_);
          if (!p.connections_.empty()) {
            std::vector<ConnectionTypeInfo> vec;
            for (auto& x : p.connections_) {
              if (!x.remoteAddresses.empty()) {
                vec.emplace_back();
                vec.back().name = connectionTypeName.at(&x - p.connections_.data());
                vec.back().addr = x.remoteAddresses;
              }
            }
            if (!vec.empty()) {
              info[p.name] = std::move(vec);
            }
          }
        }
        if (!info.empty()) {
          BufferHandle buffer;
          serializeToBuffer(buffer, rid, Rpc::reqPeerFound, info);
          log("send reqPeerFound! rid %#x\n", rid);
          conn.send(std::move(buffer), defer);
        } else {
          log("no info :(\n");
        }
      }
      break;
    }
    default:
      if (fid < (uint32_t)Rpc::reqCallOffset) {
        return;
      }
      // RPC call
      Rpc::FBase* f = nullptr;
      {
        std::lock_guard l(rpc.mutex_);
        auto i = rpc.funcs_.find(fid);
        if (i != rpc.funcs_.end()) {
          f = &*i->second;
        }
      }
      auto getFuncName = [&]() {
        std::lock_guard l(rpc.mutex_);
        for (auto& v : rpc.funcIds_) {
          if (v.second == fid) {
            return v.first;
          }
        }
        return (std::string_view) "NOT-FOUND";
      };
      if (!f) {
        BufferHandle buffer;
        serializeToBuffer(buffer, rid, Rpc::reqFunctionNotFound);
        conn.send(std::move(buffer), defer);
      } else {
        bool recvOk = handleRecv<true>(rpc.incoming_, peer, conn, rid, defer);
        if (recvOk) {
          log("got request rid %#x (%s) from %s\n", rid, getFuncName(), peer.name);
          f->call(std::move(buffer), [weak = weakLock, peer = &peer, rid](BufferHandle outbuffer) {
            auto me = weak->template lock<RpcImpl>();
            if (!me) {
              return;
            }
            Deferrer defer;
            auto& rpc = me->rpc;
            {
              auto* ptr = dataptr<std::byte>(&*outbuffer);
              std::memcpy(ptr, &rid, sizeof(rid));
              ptr += sizeof(rid);
              uint32_t outFid;
              std::memcpy(&outFid, ptr, sizeof(outFid));
              ptr += sizeof(outFid);

              SharedBufferHandle shared(outbuffer.release());
              me->log("sending response for rid %#x of %d bytes to %s\n", rid, shared->size, peer->name);

              auto now = std::chrono::steady_clock::now();
              Rpc::Impl::IncomingBucket& bucket = rpc.getBucket(rpc.incoming_, rid);
              std::lock_guard l2(rpc.incomingFifoMutex_);
              std::unique_lock l(bucket.mutex);
              size_t totalResponseSize;
              auto i = bucket.map.find(rid);
              if (i != bucket.map.end()) {
                auto& x = bucket.map[rid];
                x.responseTimestamp = now;
                totalResponseSize = rpc.totalResponseSize_ += shared->size;
                x.timeout = now + std::chrono::milliseconds(250);
                x.resend.buffer = std::move(shared);
                // log("x is %p, resend.buffer is %p\n", (void*)&x, (void*)&*x.resend.buffer);
                rpc.resend(*peer, x.resend, defer);
                listInsert(rpc.incomingFifo_.prev, &x);

                rpc.updateTimeout(now + std::chrono::seconds(1));
              } else {
                totalResponseSize = rpc.totalResponseSize_;
              }
              l.unlock();

              // Erase outgoing data if it has not been acknowledged within a
              // certain time period. This prevents us from using resources
              // for peers that are permanently gone.
              auto timeout = std::chrono::seconds(300);
              if (now - rpc.incomingFifo_.next->responseTimestamp >= std::chrono::seconds(5)) {
                if (totalResponseSize < 1024 * 1024 && rpc.incoming_.size() < 1024) {
                  timeout = std::chrono::seconds(1800);
                } else if (totalResponseSize >= 1024 * 1024 * 1024 || rpc.incoming_.size() >= 1024 * 1024) {
                  timeout = std::chrono::seconds(60);
                }
                while (rpc.incomingFifo_.next != &rpc.incomingFifo_ &&
                       now - rpc.incomingFifo_.next->responseTimestamp >= timeout) {
                  Rpc::Impl::Incoming* i = rpc.incomingFifo_.next;
                  listErase(i);
                  auto& iBucket = rpc.getBucket(rpc.incoming_, i->rid);
                  std::lock_guard l3(iBucket.mutex);
                  if (i->resend.buffer) {
                    rpc.totalResponseSize_ -= i->resend.buffer->size;
                  }
                  i->peer->addRecentIncoming(i->rid, now + std::chrono::minutes(1));
                  me->rpc.cleanup(*i, defer);
                  iBucket.map.erase(i->rid);
                  me->log("permanent timeout of response for peer %s rid %#x!?\n", i->peer->name, i->rid);
                }
              }
            }
          });
        }
      }
    }
  }

  void onResponse(
      PeerImpl& peer, RpcConnectionImpl<API>& conn, uint32_t rid, uint32_t fid, const std::byte* ptr, size_t len,
      BufferHandle buffer, Deferrer& defer) noexcept {
    log("onResponse peer %s rid %#x fid %#x\n", peer.name, rid, fid);
    rid |= 1;
    switch (fid) {
    case Rpc::reqClose: {
      log("got reqClose from %s\n", peer.name);
      auto& x = peer.connections_.at(index<API>);
      std::lock_guard l(x.mutex);
      for (size_t i = 0; i != x.conns.size(); ++i) {
        if (&*x.conns[i] == &conn) {
          peer.throwAway(x, i);
          break;
        }
      }
      break;
    }
    case Rpc::reqPoke: {
      handlePoke(rpc.outgoing_, peer, conn, rid, defer);
      break;
    }
    case Rpc::reqAck: {
      handleAck<false>(rpc.outgoing_, peer, rid, defer);
      break;
    }
    case Rpc::reqNack: {
      handleNack(rpc.outgoing_, peer, rid, defer);
      break;
    }
    case Rpc::reqPeerFound: {
      std::unordered_map<std::string_view, std::vector<ConnectionTypeInfo>> info;
      deserializeBuffer(ptr, len, info);
      for (auto& [name, vec] : info) {
        log("Received some connection info about peer %s\n", name);
        PeerImpl& peer = rpc.getPeer(name);
        std::lock_guard l(peer.idMutex_);
        for (auto& n : vec) {
          for (size_t i = 0; i != peer.connections_.size(); ++i) {
            if (connectionTypeName[i] == n.name) {
              auto& x = peer.connections_[i];
              std::lock_guard l(x.mutex);
              x.valid = true;
              for (auto& v2 : n.addr) {
                if (std::find(x.remoteAddresses.begin(), x.remoteAddresses.end(), v2) == x.remoteAddresses.end()) {
                  log("Adding address %s\n", v2);
                  x.remoteAddresses.push_back(rpc.persistentString(v2));
                  if (x.remoteAddresses.size() > 48) {
                    x.remoteAddresses.erase(
                        x.remoteAddresses.begin(), x.remoteAddresses.begin() + (x.remoteAddresses.size() - 24));
                  }
                }
              }
              if (x.remoteAddresses.size() > 24) {
                x.remoteAddresses.erase(
                    x.remoteAddresses.begin(), x.remoteAddresses.begin() + (x.remoteAddresses.size() - 24));
              }
            }
          }
        }
      }
      break;
    }
    case Rpc::reqFunctionNotFound:
    case Rpc::reqError:
    case Rpc::reqSuccess: {
      bool recvOk = handleRecv<false>(rpc.outgoing_, peer, conn, rid, defer);
      if (recvOk) {
        Rpc::ResponseCallback response;
        uint32_t ofid;
        {
          auto& oBucket = rpc.getBucket(rpc.outgoing_, rid);
          std::lock_guard l(oBucket.mutex);
          auto i = oBucket.map.find(rid);
          if (i != oBucket.map.end() && i->second.peer == &peer) {
            // log("got response for rid %#x from %s\n", rid, peer.name);
            response = std::move(i->second.response);
            ofid = i->second.fid;
            rpc.cleanup(i->second, defer);
            oBucket.map.erase(i);
          } else {
            // log("got response for unknown rid %#x from %s\n", rid, peer.name);
          }
        }
        if (response) {
          if (fid == Rpc::reqFunctionNotFound) {
            Error err("Remote function not found");
            std::move(response)(std::move(buffer), &err);
          } else if (fid == Rpc::reqError) {
            uint32_t xrid, xfid;
            std::string_view str;
            deserializeBuffer(std::move(buffer), xrid, xfid, str);
            Error err{fmt::sprintf("Remote exception during RPC call (%s): %s", peer.functionName(ofid), str)};
            std::move(response)(nullptr, &err);
          } else if (fid == Rpc::reqSuccess) {
            std::move(response)(std::move(buffer), nullptr);
          }
        }
      }
      break;
    }
    default:
      log("onResponse: unknown fid %#x\n", fid);
    }
  }

  template<typename... Args>
  void log(const char* fmt, Args&&... args) {
    rpc.log(fmt, std::forward<Args>(args)...);
  }
};

template<typename... Args>
void PeerImpl::log(const char* fmt, Args&&... args) {
  rpc.log(fmt, std::forward<Args>(args)...);
}

std::string_view PeerImpl::rpcName() {
  return rpc.myName;
}

void PeerImpl::findPeer() {
  rpc.findPeer(name);
}

void PeerImpl::throwAway(std::unique_ptr<RpcConnectionImplBase> c) {
  std::lock_guard l(rpc.garbageMutex_);
  rpc.garbageConnections_.push_back(std::move(c));
}

template<typename API, bool explicit_>
void PeerImpl::connect(std::string_view addr, Deferrer& defer) {
  rpc.connect<API, explicit_>(addr, defer);
}

Rpc::Rpc() {
  impl_ = std::make_unique<Rpc::Impl>();
}
Rpc::~Rpc() {
  close();
}
void Rpc::close() {
  impl_.reset();
  // Wait for callbacks to finish. Any remaining callbacks after impl_.reset() should not depend on impl_.
  // This is mostly to provide a guarantee that when Rpc is destructed, there are no running callbacks
  while (activeOps.load()) {
    std::this_thread::yield();
  }
}

std::pair<std::string_view, std::string_view> splitUri(std::string_view uri) {
  const char* s = uri.begin();
  const char* e = uri.end();
  const char* i = s;
  while (i != e) {
    if (*i == ':') {
      ++i;
      if (i != e && i + 1 != e && *i == '/' && i[1] == '/') {
        return {std::string_view(uri.begin(), i - 1 - s), std::string_view(i + 2, e - (i + 2))};
      } else {
        return {std::string_view(uri.begin(), i - 1 - s), std::string_view(i, e - i)};
      }
    }
    if (!std::islower(*i)) {
      break;
    }
    ++i;
  }
  return {std::string_view(), std::string_view(s, e - s)};
}

void Rpc::listen(std::string_view addr) {
  auto [scheme, path] = splitUri(addr);
  if (!scheme.empty()) {
    switchOnScheme(scheme, [&, path = path](auto api) { impl_->listen<decltype(api)>(path); });
  } else {
    impl_->listen<API_TPUV>(addr);
  }
}
void Rpc::connect(std::string_view addr) {
  Deferrer defer;
  auto [scheme, path] = splitUri(addr);
  if (!scheme.empty()) {
    switchOnScheme(scheme, [&, path = path](auto api) { impl_->connect<decltype(api)>(path, defer); });
  } else {
    impl_->connect<API_TPUV>(addr, defer);
  }
}
void Rpc::setName(std::string_view name) {
  impl_->setName(name);
}
std::string_view Rpc::getName() const {
  return impl_->myName;
}
void Rpc::sendRequest(
    std::string_view peerName, std::string_view functionName, BufferHandle buffer, ResponseCallback response) {
  impl_->sendRequest(peerName, functionName, std::move(buffer), std::move(response));
}
void Rpc::define(std::string_view name, std::unique_ptr<FBase>&& f) {
  impl_->define(name, std::move(f));
}

void Rpc::debugInfo() {
  impl_->debugInfo();
}

void Rpc::setTimeout(std::chrono::milliseconds milliseconds) {
  impl_->setTimeout(milliseconds);
}

std::chrono::milliseconds Rpc::getTimeout() {
  return impl_->getTimeout();
}

void Rpc::setTransports(const std::vector<std::string>& names) {
  impl_->setTransports(names);
}

} // namespace rpc
