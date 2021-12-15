/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "pyutil.h"
#include "synchronization.h"
#include "util.h"

#include <atomic>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace moolib {

template<typename T>
struct ResourceContainer;

template<typename T>
struct ResourceObject {
  ResourceContainer<T>* container = nullptr;
  std::atomic_size_t refCount = 0;
  static_assert(decltype(refCount)::is_always_lock_free);
  std::string name;
};

template<typename T>
struct ResourceHandle {
  std::shared_ptr<T> value = nullptr;
  ResourceHandle() = default;
  ResourceHandle(std::nullptr_t){};
  ResourceHandle(std::shared_ptr<T> value) : value(std::move(value)) {
    addRef();
  }
  ~ResourceHandle() {
    decRef();
  }
  ResourceHandle(const ResourceHandle& n) {
    value = n.value;
    addRef();
  }
  ResourceHandle(ResourceHandle&& n) {
    value = std::exchange(n.value, nullptr);
  }
  ResourceHandle& operator=(const ResourceHandle& n) {
    if (n.value != value) {
      decRef();
      value = n.value;
      addRef();
    }
    return *this;
  }
  ResourceHandle& operator=(ResourceHandle&& n) {
    std::swap(value, n.value);
    return *this;
  }
  void reset() {
    decRef();
  }
  void addRef() {
    if (value) {
      value->refCount.fetch_add(1, std::memory_order_relaxed);
    }
  }
  void decRef() {
    if (value) {
      if (value->refCount.fetch_sub(1) == 1) {
        value->container->remove(value);
        value.reset();
      }
    }
  }
  operator T&() const {
    return *value;
  }
  T& operator*() const {
    return *value;
  }
  T* operator->() const {
    return &*value;
  }
  operator bool() const {
    return value != nullptr;
  }
};

template<typename T>
struct is_shared_ptr : std::false_type {};
template<typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
template<typename T>
static constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;

template<typename T>
struct ResourceContainer {
  ResourceContainer() = default;
  ResourceContainer(const ResourceContainer&) = delete;
  ResourceContainer(ResourceContainer&&) = delete;
  std::mutex mutex;
  std::unordered_map<std::string_view, std::shared_ptr<T>> map;
  template<typename... Args>
  ResourceHandle<T> emplaceHandle(std::string_view name, Args&&... args) {
    std::lock_guard l(mutex);
    auto i = map.find(name);
    if (i != map.end() && i->second->refCount.load(std::memory_order_relaxed) > 0 && !i->second->done()) {
      return i->second;
    }
    auto obj = std::make_shared<T>(std::forward<Args>(args)...);
    obj->name = name;
    obj->container = this;
    if (i != map.end()) {
      i = map.erase(i);
      i = map.emplace_hint(i, obj->name, std::move(obj));
    } else {
      i = map.emplace(obj->name, std::move(obj)).first;
    }
    return i->second;
  }

  template<typename R>
  R find(std::string_view name) {
    std::lock_guard l(mutex);
    auto i = map.find(name);
    if (i != map.end() && i->second->refCount.load(std::memory_order_relaxed) > 0) {
      return i->second;
    } else {
      return nullptr;
    }
  }

  ResourceHandle<T> findHandle(std::string_view name) {
    return find<ResourceHandle<T>>(name);
  }
  std::shared_ptr<T> findPointer(std::string_view name) {
    return find<std::shared_ptr<T>>(name);
  }

  void remove(std::shared_ptr<ResourceObject<T>> value) {
    std::lock_guard l(mutex);
    auto i = map.find(value->name);
    if (i != map.end() && &*value == &*i->second && value->refCount.load(std::memory_order_relaxed) == 0) {
      map.erase(i);
    }
  }

  size_t size() {
    std::lock_guard l(mutex);
    return map.size();
  }
};

struct AllReduceOperation;

struct GroupInfo : ResourceObject<GroupInfo> {
  std::mutex mutex;
  std::string brokerName;
  std::vector<std::string> members;
  std::atomic_bool wantsResync = false;
  std::atomic_bool isResyncing = false;
  std::atomic_bool resyncError = false;
  bool haveUpdate = false;
  std::atomic_uint32_t syncId = 0;
  uint32_t newSyncId = 0;
  std::vector<std::string> newMembers;
  int32_t sortOrder = 0;

  Future<uint32_t> pingFuture;
  bool hasPinged = false;
  std::chrono::steady_clock::time_point lastPing;
  std::chrono::steady_clock::time_point lastPingResponse;
  bool brokerConnectionIsActive = false;

  std::vector<std::weak_ptr<AllReduceOperation>> activeAllReductions;

  GroupInfo(std::string brokerName) : brokerName(std::move(brokerName)) {}

  bool done() const noexcept {
    return false;
  }
};

struct AccumulatorReductionType {
  std::vector<rpc::Tensor> gradients;
  size_t numGradients = 0;
  size_t numSkipped = 0;
  size_t batchSize = 0;

  void add(AccumulatorReductionType& n) {
    if (n.gradients.size() == gradients.size()) {
      for (size_t i = 0; i != gradients.size(); ++i) {
        gradients[i] += n.gradients[i];
      }
    } else if (n.gradients.size() > gradients.size()) {
      std::swap(gradients, n.gradients);
    }
    numGradients += n.numGradients;
    numSkipped += n.numSkipped;
    batchSize += n.batchSize;
  }

  template<typename X>
  void serialize(X& x) {
    x(gradients, numGradients, numSkipped, batchSize);
  }
};

struct AccumulatorFindLeaderType {
  int64_t modelVersion;
  std::string name;

  template<typename X>
  void serialize(X& x) {
    x(modelVersion, name);
  }
};

using ReduceVariant = std::variant<
    rpc::Tensor, std::vector<rpc::Tensor>, GilWrapper<py::object>, AccumulatorFindLeaderType, AccumulatorReductionType,
    size_t>;

struct BuiltinOp {
  template<typename T, typename T2>
  void operator()(T&&, T2&&) {
    fatal(
        "Builtin reduce operators can only be applied to tensors! (got %s and %s)", typeid(T).name(),
        typeid(T2).name());
  }
};

struct ReduceSum : BuiltinOp {
  void operator()(rpc::Tensor& local, rpc::Tensor& remote) {
    local += remote;
  }
};
struct ReduceProduct : BuiltinOp {
  void operator()(rpc::Tensor& local, rpc::Tensor& remote) {
    local *= remote;
  }
};
struct ReduceMin : BuiltinOp {
  void operator()(rpc::Tensor& local, rpc::Tensor& remote) {
    rpc::min_out(local, local, remote);
  }
};
struct ReduceMax : BuiltinOp {
  void operator()(rpc::Tensor& local, rpc::Tensor& remote) {
    rpc::max_out(local, local, remote);
  }
};

struct AllReduceOperation : ResourceObject<AllReduceOperation> {
  struct PeerInfo {
    std::string name;
  };

  ~AllReduceOperation() {
    rpc::FunctionPointer func = callback.load(std::memory_order_relaxed);
    if (func) {
      (rpc::Function<void(rpc::Tensor*, rpc::Error*)>(func));
    }
  }

  bool done() const noexcept {
    return flags.load(std::memory_order_relaxed) != 0;
  }

  std::atomic_bool starting = false;
  std::atomic_bool started = false;
  std::vector<PeerInfo> peers;
  uint32_t syncId = 0;
  ReduceVariant localData;
  ReduceVariant result;
  rpc::Function<void(ReduceVariant&, ReduceVariant&)> op;

  size_t myPeerIndex = 0;

  std::mutex opMutex;
  std::atomic_bool hasSent = false;
  std::array<std::atomic_bool, 2> hasReceived{};

  std::chrono::steady_clock::time_point timestamp;
  std::shared_ptr<GroupInfo> group;

  std::atomic_int flags = 0;

  rpc::SpinMutex errorMutex;
  std::optional<rpc::Error> error;

  std::atomic<rpc::FunctionPointer> callback = nullptr;
  void doCallback() {
    if (callback.load(std::memory_order_relaxed)) {
      rpc::FunctionPointer func = callback.exchange(nullptr);
      if (func) {
        int f = flags.load(std::memory_order_relaxed);
        if (f & 1) {
          (rpc::Function<void(ReduceVariant*, rpc::Error*)>(func))(&result, nullptr);
        } else if (f & 2) {
          std::lock_guard l(errorMutex);
          (rpc::Function<void(ReduceVariant*, rpc::Error*)>(func))(nullptr, &*error);
        } else {
          fatal("internal error: AllReduceOperation unhandled state");
        }
      }
    }
  }

  void setException(rpc::Error error) {
    {
      std::lock_guard l(errorMutex);
      this->error = error;
      flags |= 2;
    }
    doCallback();
  }
};

struct GroupService {

  rpc::Rpc* rpc = nullptr;

  ResourceContainer<GroupInfo> groups;

  GroupService(rpc::Rpc& rpc) : rpc(&rpc) {
    setup();
  }
  ~GroupService() {}

  void setup() {

    rpc->define<std::pair<uint32_t, int32_t>(std::string_view, uint32_t)>(
        "GroupService::sync", [this](std::string_view groupName, uint32_t syncId) {
          auto group = groups.findPointer(groupName);
          if (group) {
            std::lock_guard l(group->mutex);
            group->isResyncing = true;
            group->haveUpdate = false;
            group->newSyncId = syncId;
            log.debug("got sync request %s::%#x\n", groupName, syncId);
            return std::make_pair(syncId, group->sortOrder);
          } else {
            return std::make_pair((uint32_t)-1, (int32_t)-1);
          }
        });

    rpc->define<void(std::string_view, uint32_t, std::vector<std::string>)>(
        "GroupService::update", [this](std::string_view groupName, uint32_t syncId, std::vector<std::string> members) {
          auto group = groups.findPointer(groupName);
          if (group && syncId == group->newSyncId) {
            std::lock_guard l(group->mutex);
            log.debug("group %s::%#x update %d members\n", groupName, syncId, members.size());
            group->newMembers = std::move(members);
            group->haveUpdate = true;
          }
        });
  }

  void resync(GroupInfo& group) {
    if (group.wantsResync.load(std::memory_order_relaxed) | group.isResyncing.load(std::memory_order_relaxed)) {
      return;
    }
    group.wantsResync.store(true, std::memory_order_relaxed);
    rpc->asyncCallback<void>(
        group.brokerName, "BrokerService::resync",
        [this, groupName = group.name](rpc::Error* e) {
          log.error("resync error: %s", e->what());
          auto h = groups.findHandle(groupName);
          if (h) {
            h->resyncError.store(true, std::memory_order_relaxed);
          }
        },
        group.name);
  }

  bool update(GroupInfo& group, uint32_t sortOrder, uint32_t timeoutMilliseconds) {
    auto now = std::chrono::steady_clock::now();

    auto pingNow = [&]() {
      group.lastPing = now;
      group.hasPinged = true;
      group.pingFuture = callImpl<uint32_t>(
          *rpc, group.brokerName, "BrokerService::ping", group.name, rpc->getName(), timeoutMilliseconds);
    };

    std::lock_guard l(group.mutex);
    group.sortOrder = sortOrder;

    bool updated = group.isResyncing && group.haveUpdate;

    if (updated) {
      group.wantsResync = false;
      group.isResyncing = false;
      group.haveUpdate = false;
      group.syncId = group.newSyncId;
      group.members = std::move(group.newMembers);

      pingNow();
    } else if (group.wantsResync && !group.isResyncing) {
      if (group.resyncError) {
        log.verbose("Error during resync, trying again");
        group.resyncError = false;
        resync(group);
      }
    }

    if (!group.hasPinged || updated ||
        now - group.lastPing >=
            std::min(std::chrono::milliseconds(4000), std::chrono::milliseconds(timeoutMilliseconds) / 2)) {
      if (group.hasPinged && !group.pingFuture) {
        if (now - group.lastPingResponse >= std::chrono::seconds(8)) {
          log.verbose("Broker has not responded in %g seconds!\n", seconds(now - group.lastPingResponse));
        }
        if (group.brokerConnectionIsActive &&
            now - group.lastPingResponse >= std::chrono::seconds(1) + std::chrono::milliseconds(timeoutMilliseconds)) {
          group.brokerConnectionIsActive = false;
          group.members.clear();
          group.syncId = 0;
          resync(group);
          updated = true;
        }
      } else {
        group.brokerConnectionIsActive = true;
        group.lastPingResponse = now;
      }
      if (group.pingFuture && *group.pingFuture != group.syncId) {
        group.members.clear();
        group.syncId = 0;
        resync(group);
        updated = true;
      }
      pingNow();
    }

    if (updated) {
      for (auto& wh : group.activeAllReductions) {
        auto h = wh.lock();
        if (h && h->flags.load(std::memory_order_relaxed) == 0) {
          h->setException(rpc::Error("AllReduce operation cancelled due to a group change"));
        }
      }
      group.activeAllReductions.clear();
    } else {
      auto timeout = rpc->getTimeout();
      auto now = std::chrono::steady_clock::now();
      bool shouldResync = false;
      group.activeAllReductions.erase(
          std::remove_if(
              group.activeAllReductions.begin(), group.activeAllReductions.end(),
              [&](auto& wh) {
                auto h = wh.lock();
                if (!h) {
                  return true;
                }
                if (h->flags.load(std::memory_order_relaxed)) {
                  return true;
                }
                if (now >= h->timestamp + timeout) {
                  shouldResync = true;
                  h->setException(rpc::Error("AllReduce operation timed out"));
                  return true;
                }
                return false;
              }),
          group.activeAllReductions.end());
      if (shouldResync) {
        resync(group);
      }
    }

    return updated;
  }
};

struct AllReduce {
  std::shared_ptr<rpc::Rpc> rpc;
  ResourceHandle<AllReduceOperation> h;
  AllReduce() = default;
  AllReduce(std::shared_ptr<rpc::Rpc> rpc, ResourceHandle<AllReduceOperation> h)
      : rpc(std::move(rpc)), h(std::move(h)) {}
};

template<typename T>
struct is_variant : std::false_type {};
template<typename T>
struct is_variant<std::variant<T>> : std::true_type {};
template<typename T>
static constexpr bool is_variant_v = is_variant<T>::value;

struct AllReduceService {

  rpc::Rpc* rpc = nullptr;
  GroupService* groupService = nullptr;

  std::mutex mutex;

  ResourceContainer<AllReduceOperation> ops;

  struct QueueEntry {
    std::chrono::steady_clock::time_point timestamp;
    std::string name;
    uint32_t syncId;
    size_t sourcePeerIndex;
    ReduceVariant data;
  };

  std::mutex queueMutex;
  std::vector<QueueEntry> queue;

  AllReduceService(rpc::Rpc& rpc) : rpc(&rpc) {
    groupService = this->rpc->getService<GroupService>("GroupService");
    setup();
  }

  template<typename F, typename Op, typename... Args>
  decltype(auto) visit(F&& f, Op&& op, Args&&... args) {
    if constexpr (is_variant_v<Op>) {
      return std::visit(std::forward<F>(f), std::forward<Op>(op), std::forward<Args>(args)...);
    } else {
      return std::visit(
          [&](auto&&... args) { std::forward<F>(f)(std::forward<Op>(op), std::forward<decltype(args)>(args)...); },
          std::forward<Args>(args)...);
    }
  }

  void sendShare(AllReduceOperation* h, size_t end) {
    size_t myPeerIndex = h->myPeerIndex;
    size_t begin = myPeerIndex + 1;
    if (begin >= end) {
      return;
    }
    size_t mid = begin + (end - begin) / 2;
    log.debug("%d share with [%d, %d)\n", myPeerIndex, begin, mid);
    rpc->asyncCallback<void>(
        h->peers[begin].name, "AllReduceService::share", nullptr, h->name, h->syncId, mid, h->result);
    if (mid != begin) {
      log.debug("%d share with [%d, %d)\n", myPeerIndex, mid, end);
      rpc->asyncCallback<void>(
          h->peers[mid].name, "AllReduceService::share", nullptr, h->name, h->syncId, end, h->result);
    }
  }

  bool reduce(std::string_view name, uint32_t syncId, size_t sourcePeerIndex, ReduceVariant& data) noexcept {
    std::shared_ptr<AllReduceOperation> h = ops.findPointer(name);
    if (!h) {
      log.debug("reduce: null h\n");
    } else if (!h->started) {
      log.debug("reduce: not started\n");
    } else if (h->syncId != syncId) {
      log.debug("reduce: syncId mismatch\n");
    }
    if (h && h->started && h->syncId == syncId) {
      if (h->group->syncId != syncId) {
        return false;
      }
      size_t myPeerIndex = h->myPeerIndex;
      log.debug("%d reduce recv from %d\n", myPeerIndex, sourcePeerIndex);
      int receiveIndex = sourcePeerIndex - myPeerIndex * 2;
      if (receiveIndex != 0 && receiveIndex != 1) {
        return false;
      }
      if (h->localData.index() != data.index()) {
        return false;
      }
      if (h->hasReceived[receiveIndex].load(std::memory_order_relaxed)) {
        return false;
      }
      bool done;
      {
        std::lock_guard l(h->opMutex);
        if (h->hasReceived[receiveIndex].exchange(true, std::memory_order_relaxed)) {
          return false;
        }
        h->op(h->localData, data);
        done = h->hasReceived[receiveIndex ^ 1].load(std::memory_order_relaxed);
      }

      if (done) {
        if (!h->hasSent.load(std::memory_order_relaxed) && !h->hasSent.exchange(true, std::memory_order_relaxed)) {
          if (myPeerIndex == 0) {
            log.debug("receive done, enter share mode!");
            h->result = std::move(h->localData);
            sendShare(&*h, h->peers.size());
            h->flags |= 1;
            h->doCallback();
          } else {
            log.debug("receive done, pass on!");
            rpc->asyncCallback<void>(
                h->peers[myPeerIndex / 2].name, "AllReduceService::reduce",
                [h](rpc::Error* error) {
                  h->setException(std::move(*error));
                  h->doCallback();
                },
                h->name, h->syncId, myPeerIndex, h->localData);
          }
        }
      }
      return true;
    } else {
      return false;
    }
  }

  void share(std::string_view name, uint32_t syncId, size_t end, ReduceVariant& data) noexcept {
    std::shared_ptr<AllReduceOperation> h = ops.findPointer(name);
    if (!h) {
      log.debug("share: null h\n");
    } else if (!h->started) {
      log.debug("share: not started\n");
    } else if (h->syncId != syncId) {
      log.debug("share: syncId mismatch\n");
    }
    if (h && h->started && h->syncId == syncId) {
      if (h->group->syncId != syncId) {
        return;
      }
      if (h->localData.index() != data.index()) {
        log.debug("index mismatch\n");
        return;
      }
      log.debug("got share [%d, %d)\n", h->myPeerIndex, end);
      h->result = std::move(data);
      sendShare(&*h, end);
      h->flags |= 1;
      h->doCallback();
    }
  }

  void setup() {
    rpc->define<void(std::string_view, uint32_t, size_t, ReduceVariant)>(
        "AllReduceService::reduce",
        [this](std::string_view name, uint32_t syncId, size_t sourcePeerIndex, ReduceVariant data) {
          log.debug("%s recv reduce %d\n", rpc->getName(), sourcePeerIndex);
          if (!reduce(name, syncId, sourcePeerIndex, data)) {
            std::lock_guard l(queueMutex);
            if (!reduce(name, syncId, sourcePeerIndex, data)) {
              log.debug("adding to queue\n");
              queue.emplace_back();
              QueueEntry& e = queue.back();
              e.timestamp = std::chrono::steady_clock::now();
              e.name = name;
              e.syncId = syncId;
              e.sourcePeerIndex = sourcePeerIndex;
              e.data = std::move(data);
            }
          }
        });
    rpc->define<void(std::string_view, uint32_t, size_t, ReduceVariant)>(
        "AllReduceService::share", [this](std::string_view name, uint32_t syncId, size_t end, ReduceVariant data) {
          log.debug("%s recv share %d\n", rpc->getName(), end);
          share(name, syncId, end, data);
        });
  }

  template<typename Data, typename Op, typename Callback>
  ResourceHandle<AllReduceOperation>
  allReduce(ResourceHandle<GroupInfo> group, std::string name, Data&& data, Op&& op, Callback&& callback) {

    std::string opName = fmt::sprintf("%#x.%s::%s", group->syncId.load(), group->name, name);
    log.verbose("Doing allReduce %s\n", opName);

    ResourceHandle<AllReduceOperation> h = ops.emplaceHandle(opName);

    if (h->starting.exchange(true)) {
      throw std::runtime_error("Attempt to all-reduce twice concurrently with the name '" + name + "'");
    }

    h->timestamp = std::chrono::steady_clock::now();

    {
      std::lock_guard l(group->mutex);

      auto& members = group->members;
      auto i = std::find(members.begin(), members.end(), rpc->getName());
      if (i == members.end()) {
        throw std::runtime_error("AllReduce: local peer is not a member of the specified group!");
      }

      size_t myPeerIndex = i - members.begin();

      h->myPeerIndex = myPeerIndex;

      h->syncId = group->syncId;
      h->peers = std::vector<AllReduceOperation::PeerInfo>(members.size());
      for (size_t i = 0; i != members.size(); ++i) {
        h->peers[i].name = members[i];
      }
      h->group = group.value;
      h->group->activeAllReductions.push_back(h.value);
    }

    h->op = [op = std::forward<Op>(op)](ReduceVariant& local, ReduceVariant& remote) mutable {
      op(std::get<std::decay_t<Data>>(local), std::get<std::decay_t<Data>>(remote));
    };
    h->localData = std::move(data);

    h->callback =
        rpc::Function<void(ReduceVariant*, rpc::Error*)>([callback = std::forward<Callback>(callback), h = h.value](
                                                             ReduceVariant* r, rpc::Error* error) mutable noexcept {
          auto now = std::chrono::steady_clock::now();
          log.verbose("AllReduce %s completed in %g\n", h->name, seconds(now - h->timestamp));
          callback(r ? &std::get<std::decay_t<Data>>(*r) : nullptr, error);
        }).release();

    if (!(group->wantsResync | group->isResyncing)) {
      if (h->peers.size() == 1) {
        h->result = std::move(h->localData);
        h->flags |= 1;
        h->doCallback();
      } else {

        size_t myPeerIndex = h->myPeerIndex;
        if (myPeerIndex == 0) {
          h->hasReceived[0].store(true, std::memory_order_relaxed);
          h->started = true;
        } else {
          size_t childIndex = myPeerIndex * 2;
          if (childIndex >= h->peers.size()) {
            h->hasReceived[0].store(true, std::memory_order_relaxed);
            h->hasReceived[1].store(true, std::memory_order_relaxed);
            h->started = true;
            rpc->asyncCallback<void>(
                h->peers[myPeerIndex / 2].name, "AllReduceService::reduce",
                [h](rpc::Error* error) {
                  h->setException(std::move(*error));
                  h->doCallback();
                },
                h->name, h->syncId, myPeerIndex, h->localData);
          } else if (childIndex + 1 == h->peers.size()) {
            h->hasReceived[1].store(true, std::memory_order_relaxed);
            h->started = true;
          } else {
            h->started = true;
          }
        }
      }

      {
        auto now = std::chrono::steady_clock::now();
        auto timeout = rpc->getTimeout();
        glock l(queueMutex);
        log.debug("de-queue %d\n", queue.size());
        queue.erase(
            std::remove_if(
                queue.begin(), queue.end(),
                [&](auto& v) {
                  return reduce(v.name, v.syncId, v.sourcePeerIndex, v.data) || now >= v.timestamp + timeout;
                }),
            queue.end());
      }
    }

    return h;
  }
};

struct Group {
  std::shared_ptr<rpc::Rpc> rpc;
  GroupService* groupService = nullptr;
  AllReduceService* allReduceService = nullptr;

  std::string brokerName = "broker";
  std::string groupName;
  uint32_t timeoutMilliseconds = 10 * 1000;
  int32_t sortOrder = 0;

  ResourceHandle<GroupInfo> group;

  Group(std::shared_ptr<rpc::Rpc> rpc, std::string groupName) : rpc(std::move(rpc)), groupName(groupName) {
    groupService = this->rpc->getService<GroupService>("GroupService");
    allReduceService = this->rpc->getService<AllReduceService>("AllReduceService");
    group = groupService->groups.emplaceHandle(groupName, brokerName);
  }

  void setBrokerName(std::string brokerName) {
    this->brokerName = brokerName;
  }

  void setTimeout(float timeoutSeconds) {
    timeoutMilliseconds = timeoutSeconds * 1000;
  }

  void setSortOrder(int32_t sortOrder) {
    this->sortOrder = sortOrder;
  }

  bool update() {
    if (groupService->update(*group, sortOrder, timeoutMilliseconds)) {
      return true;
    }
    return false;
  }

  bool active() const {
    return !group->members.empty();
  }

  uint32_t syncId() const {
    return group->syncId;
  }
  const std::vector<std::string>& members() const {
    return group->members;
  }
  std::string_view name() const {
    return group->name;
  }

  template<typename Data, typename Op, typename Callback>
  AllReduce allReduce(std::string name, Data&& data, Op&& op, Callback&& callback) {
    auto h = allReduceService->allReduce(
        group, std::move(name), std::forward<Data>(data), std::forward<Op>(op), std::forward<Callback>(callback));
    return AllReduce(rpc, std::move(h));
  }
};

} // namespace moolib
