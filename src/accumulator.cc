/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "logging.h"
#include "pythonserialization.h"

#include "group.h"

#include "accumulator.h"

namespace moolib {

struct AccumulatorResource : ResourceObject<AccumulatorResource> {
  std::mutex mutex;
  uint32_t syncId = 0;
  std::vector<std::string> requestedModelUpdate;
  int64_t modelVersion = 0;
  int64_t newModelVersion = 0;
  bool haveNewParameters = false;
  bool haveNewBuffers = false;
  std::vector<rpc::Tensor> newParameters;
  std::vector<rpc::Tensor> newBuffers;
  GilWrapper<py::object> newUserState;
  std::vector<rpc::Tensor> modelParameters;
  std::vector<rpc::Tensor> modelBuffers;

  bool done() const noexcept {
    return false;
  }
};

struct AccumulatorService {
  rpc::Rpc* rpc = nullptr;
  AccumulatorService(rpc::Rpc& rpc) : rpc(&rpc) {
    setup();
  }
  ~AccumulatorService() {}

  ResourceContainer<AccumulatorResource> resources;

  void setup() {

    rpc->define<bool(std::string_view, uint32_t, std::string_view)>(
        "AccumulatorService::requestModel",
        [this](std::string_view resName, uint32_t syncId, std::string_view peerName) {
          std::shared_ptr<AccumulatorResource> h = resources.findPointer(resName);
          if (h) {
            std::unique_lock l(h->mutex);
            if (syncId != h->syncId) {
              log.debug("Got model update for wrong syncId (%#x, should be %#x)\n", syncId, h->syncId);
              return false;
            }
            if (std::find(h->requestedModelUpdate.begin(), h->requestedModelUpdate.end(), peerName) ==
                h->requestedModelUpdate.end()) {
              h->requestedModelUpdate.push_back(std::string(peerName));
            }
            return true;
          }
          return false;
        });

    rpc->define<bool(
        std::string_view, uint32_t, bool, int64_t, std::vector<rpc::Tensor>, std::vector<rpc::Tensor>,
        GilWrapper<py::object>)>(
        "AccumulatorService::modelUpdate",
        [this](
            std::string_view resName, uint32_t syncId, bool isRegularUpdate, int64_t modelVersion,
            std::vector<rpc::Tensor> parameters, std::vector<rpc::Tensor> buffers, GilWrapper<py::object> userState) {
          std::shared_ptr<AccumulatorResource> h = resources.findPointer(resName);
          if (h) {
            std::unique_lock l(h->mutex);
            if (syncId != h->syncId) {
              log.debug("Got model update for wrong syncId (%#x, should be %#x)\n", syncId, h->syncId);
              return false;
            }
            if (isRegularUpdate && modelVersion != h->modelVersion) {
              log.debug(
                  "Got regular model update for wrong modelVersion (%#x, should be %#x)\n", modelVersion,
                  h->modelVersion);
              return false;
            }
            if (parameters.size() != h->modelParameters.size()) {
              log.debug(
                  "Got model update for wrong number of parameters (%d, should be %d)\n", parameters.size(),
                  h->modelParameters.size());
              return false;
            }
            if (buffers.size() != h->modelBuffers.size()) {
              log.debug(
                  "Got model update for wrong number of buffers (%d, should be %d)\n", buffers.size(),
                  h->modelBuffers.size());
              return false;
            }
            log.debug("got modelUpdate %d\n", modelVersion);
            h->haveNewParameters = true;
            h->newModelVersion = modelVersion;
            h->newParameters = std::move(parameters);
            h->newBuffers = std::move(buffers);
            h->newUserState = std::move(*userState);
            return true;
          } else {
            return false;
          }
        });

    rpc->define<bool(std::string_view, uint32_t, std::vector<rpc::Tensor>)>(
        "AccumulatorService::buffersUpdate",
        [this](std::string_view resName, uint32_t syncId, std::vector<rpc::Tensor> buffers) {
          std::shared_ptr<AccumulatorResource> h = resources.findPointer(resName);
          if (h) {
            std::unique_lock l(h->mutex);
            if (syncId != h->syncId) {
              log.debug("Got buffers update for wrong syncId (%#x, should be %#x)\n", syncId, h->syncId);
              return false;
            }
            log.debug("Got buffers\n");
            h->haveNewBuffers = true;
            h->newBuffers = std::move(buffers);
            return true;
          } else {
            return false;
          }
        });
  }
};

struct ResultCallback {
  std::atomic<rpc::FunctionPointer> callback = nullptr;
  ~ResultCallback() {
    rpc::FunctionPointer p = callback.load();
    if (p) {
      (rpc::Function<void()>(p));
    }
  }
  ResultCallback& operator=(rpc::Function<void()> f) noexcept {
    auto prev = callback.exchange(f.release());
    if (prev) {
      (rpc::Function<void()>(prev));
    }
    return *this;
  }
  bool tryCall() noexcept {
    if (callback.load(std::memory_order_relaxed)) {
      auto f = callback.exchange(nullptr);
      if (f) {
        (rpc::Function<void()>(f))();
        return true;
      }
    }
    return false;
  }
  operator bool() const noexcept {
    return callback != nullptr;
  }
};

struct AccumulatorImpl {
  std::atomic_bool terminate_ = false;

  std::shared_ptr<rpc::Rpc> rpc;
  GroupService* groupService = nullptr;
  AllReduceService* allReduceService = nullptr;
  AccumulatorService* accumulatorService = nullptr;
  ResourceHandle<GroupInfo> group;
  std::string resName_;
  bool shouldUpdateGroup_ = false;

  ResourceHandle<AccumulatorResource> h;

  std::string syncLeader;

  bool isFindingLeader = false;
  std::vector<std::string> members_;

  bool hasReceivedModel_ = false;

  std::string myName;
  bool isWaitingForModel = false;
  std::chrono::steady_clock::time_point isWaitingForModelTimestamp;

  std::chrono::steady_clock::time_point lastSentBuffers;
  std::chrono::steady_clock::time_point lastSentModel;
  std::chrono::steady_clock::time_point lastReceivedModel;

  bool hasGradients_ = false;
  bool wantsUserState_ = false;
  bool hasNewUserState_ = false;
  GilWrapper<py::object> userState_;
  async::SchedulerFifo async{8};

  using ReductionType = AccumulatorReductionType;

  struct ReduceGradientsContainer {
    bool reduceStarted = false;
    bool reduceDone = false;
    bool isCounting = false;
    bool wantsMoreCounting = false;
    ResultCallback result;
    ResourceHandle<AllReduceOperation> reduce;
    ReductionType data;
    uint32_t syncId = 0;

    AccumulatorImpl* accumulator = nullptr;

    ReduceGradientsContainer(AccumulatorImpl* accumulator) : accumulator(accumulator) {}
    ~ReduceGradientsContainer() {
      if (!data.gradients.empty()) {
        accumulator->freeGradients(std::move(data.gradients));
      }
    }
  };

  std::shared_ptr<ResultCallback> findLeaderResult;
  ResourceHandle<AllReduceOperation> findLeaderReduce;

  size_t nextGradientReductionIndex = 0;
  size_t nextGradientReductionResultIndex = 0;
  bool isCopyingGradients = false;

  std::shared_ptr<ResultCallback> requestModelResult;

  size_t virtualBatchSize = 1;

  GradientStats gradientStats_;
  bool gradsOnCuda = false;

  void setParallelGradients(int n) {
    glock l(h->mutex);
    gradientReductions.resize(n);
    nextGradientReductionIndex = 0;
    nextGradientReductionResultIndex = 0;
  }

  void setVirtualBatchSize(int n) {
    glock l(h->mutex);
    virtualBatchSize = n;
  }

  std::string getLeader() {
    glock l(h->mutex);
    return syncLeader;
  }

  bool isLeader() {
    glock l(h->mutex);
    return syncLeader == myName;
  }

  void initialize(const Group& group, std::string name, pybind11::object parameters, pybind11::object buffers) {
    rpc = group.rpc;
    this->group = group.group;
    groupService = group.groupService;
    allReduceService = group.allReduceService;
    accumulatorService = rpc->getService<AccumulatorService>("AccumulatorService");

    resName_ = std::string(group.name()) + "::" + name;

    h = accumulatorService->resources.emplaceHandle(resName_);

    myName = rpc->getName();

    gradientReductions.resize(1);

    for (auto& v : parameters) {
      auto t = rpc::tryFromPython(v);
      if (!t) {
        fatal("Accumulator parameter is not a Tensor!");
      }
      gradsOnCuda |= t->is_cuda();
      h->modelParameters.push_back(*t);
    }
    for (auto& v : buffers) {
      auto t = rpc::tryFromPython(v);
      if (!t) {
        fatal("Accumulator buffer is not a Tensor!");
      }
      h->modelBuffers.push_back(*t);
    }

    if (gradsOnCuda && !rpc::CudaSupported()) {
      fatal("Accumulator was passed CUDA parameters, but is not built with CUDA support!");
    }

    freeGradients(allocateGradients());
  }

  AccumulatorImpl(const Group& group, std::string name, pybind11::object parameters, pybind11::object buffers) {
    initialize(group, name, parameters, buffers);
  }
  AccumulatorImpl(std::string name, pybind11::object parameters, pybind11::object buffers) {
    shouldUpdateGroup_ = true;
    initialize(Group(std::make_shared<rpc::Rpc>(), name), "accumulator", parameters, buffers);
  }
  ~AccumulatorImpl() {}

  void connect(std::string address) {
    rpc->connect(address);
  }

  struct crc32_t {
    std::array<uint32_t, 256> table;
    crc32_t() {
      for (uint32_t i = 0; i != 256; ++i) {
        uint32_t v = i;
        for (size_t b = 0; b != 8; ++b) {
          v = (v >> 1) ^ (v & 1 ? 0xedb88320 : 0);
        }
        table[i] = v;
      }
    }
    uint32_t operator()(const uint8_t* data, size_t data_size) {
      uint32_t r = 0xffffffff;
      const uint8_t* end = data + data_size;
      for (; data != end; ++data) {
        r = (r >> 8) ^ table[(r ^ *data) & 0xff];
      }
      return r;
    }
  };

  static size_t computeStorageNbytes(rpc::IntArrayRef sizes, rpc::IntArrayRef strides, size_t itemsize_bytes) {
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

  uint64_t tensorChecksum(rpc::Tensor t) {
    uint8_t* data = (uint8_t*)t.data_ptr();
    size_t len = computeStorageNbytes(t.sizes(), t.strides(), t.itemsize());
    return crc32_t{}(data, len);
  }

  std::vector<uint64_t> calculateChecksums(std::vector<rpc::Tensor> tensors) {
    std::vector<uint64_t> r;
    for (auto& t : tensors) {
      r.push_back(tensorChecksum(t));
    }
    return r;
  }

  bool connectedImpl() const {
    return !group->members.empty() & !members_.empty() & (hasReceivedModel_ || syncLeader == myName);
  }

  bool connected() const {
    glock l(h->mutex);
    return connectedImpl();
  }

  bool wantsState() const {
    glock l(h->mutex);
    return wantsUserState_;
  }
  bool hasNewState() const {
    return hasNewUserState_;
  }
  bool hasGradients() const {
    glock l(h->mutex);
    return hasGradients_;
  }
  bool wantsGradients() const {
    glock l(h->mutex);
    return connectedImpl() && wantsGradientsAtIndex(nextGradientReductionIndex) & !isWaitingForModel &
                                  !isFindingLeader & !isCopyingGradients & !hasGradients_;
  }

  bool wantsGradientsAtIndex(size_t index) const {
    auto& v = gradientReductions.at(index);
    return !v || v->reduceDone || !v->reduceStarted;
  }

  GradientStats getGradientStats() const {
    return gradientStats_;
  }

  void actuallyZeroGradients() {
    for (auto& v : h->modelParameters) {
      auto grad = v.mutable_grad();
      if (grad.defined()) {
        grad.detach_();
        grad.zero_();
      }
    }
  }

  void zeroGradients() {
    hasGradients_ = false;
    actuallyZeroGradients();
  }

  void setGradients(ReductionType& data) {
    if (data.gradients.empty()) {
      log.verbose("syncedGrads is empty!\n");
      actuallyZeroGradients();
    } else {
      log.verbose(
          "Adding in %d gradients (%d skipped) - %d total batch size\n", data.numGradients, data.numSkipped,
          data.batchSize);
      if (data.numGradients) {
        size_t i = 0;
        for (auto& v : h->modelParameters) {
          auto grad = v.mutable_grad();
          if (grad.defined()) {
            if (i == data.gradients.size()) {
              fatal("grads grew?");
            }
            grad.copy_(data.gradients[i], true);
            grad.mul_(1.0f / data.numGradients);
            ++i;
          }
        }
        if (i != data.gradients.size()) {
          fatal("grads shrank?");
        }
        if (gradsOnCuda) {
          rpc::getCurrentCUDAStream().synchronize();
        }
      }
    }
    ++h->modelVersion;
    log.debug("modelVersion is now %d\n", h->modelVersion);

    gradientStats_.numGradients = data.numGradients;
    gradientStats_.numSkipped = data.numSkipped;
    gradientStats_.batchSize = data.batchSize;

    hasGradients_ = true;
  }

  void requestModel() {
    if (syncLeader == myName) {
      return;
    }
    log.verbose("Requesting model\n");
    isWaitingForModel = true;
    isWaitingForModelTimestamp = std::chrono::steady_clock::now();
    requestModelResult = std::make_shared<ResultCallback>();
    rpc->asyncCallback<bool>(
        syncLeader, "AccumulatorService::requestModel",
        [requestModelResult = this->requestModelResult, this](bool* r, rpc::Error* error) {
          if (r) {
            if (*r) {
              log.debug("requestModel returned success\n");
              return;
            } else {
              log.error("requestModel returned failure\n");
            }
          } else {
            log.error("requestModel RPC failed; %s", error->what());
          }
          *requestModelResult = [this]() { onError(); };
        },
        resName_, h->syncId, myName);
  };

  void onError() {
    // If the error was not because of a resync, a resync will be necessary
    // to reset and get everything into a synchronized state again.
    if (h->syncId == group->syncId) {
      resync();
    }
  }

  void resync() {
    if (!members_.empty()) {
      log.debug("Requesting resync\n");
    }

    groupService->resync(*group);
  }

  std::chrono::steady_clock::time_point lastlog = std::chrono::steady_clock::now();

  void update() {
    rpc::AutoGradMode ng(false);

    if (shouldUpdateGroup_) {
      groupService->update(*group, 0, 10 * 1000);
    }

    glock l(h->mutex);

    auto now = std::chrono::steady_clock::now();

    bool logit = now - lastlog >= std::chrono::seconds(5);
    if (logit) {
      lastlog = now;
      log.debug("DEBUG syncUpdate()\n");
    }

    auto result = [&](auto&& r) {
      if (r) {
        if (r->tryCall()) {
          r = {};

          return true;
        }
      }
      return false;
    };

    if (result(findLeaderResult)) {
      if (syncLeader == myName) {
        log.verbose("I am now the leader!\n");
      }
    }
    if (!hasGradients_ && !isCopyingGradients) {
      auto& v = gradientReductions[nextGradientReductionResultIndex];
      if (v) {
        result(&v->result);
        if (nextGradientReductionResultIndex == gradientReductions.size() - 1) {
          nextGradientReductionResultIndex = 0;
        } else {
          ++nextGradientReductionResultIndex;
        }
      }
    }

    result(requestModelResult);

    if (h->syncId != group->syncId) {
      h->syncId = group->syncId;
      syncLeader.clear();

      findLeaderResult = {};
      for (auto& v : gradientReductions) {
        v = {};
      }
      nextGradientReductionIndex = 0;
      nextGradientReductionResultIndex = 0;
      requestModelResult = {};
      h->requestedModelUpdate.clear();

      h->haveNewParameters = false;
      hasNewUserState_ = false;
      wantsUserState_ = false;

      isFindingLeader = true;
      isWaitingForModel = false;
      hasGradients_ = false;
      members_.clear();
      if (h->syncId == 0) {
        log.verbose("Sync failure, could not join group\n", h->syncId);
      } else {
        log.verbose("Sync %#x success, finding leader\n", h->syncId);

        findLeaderResult = std::make_shared<ResultCallback>();
        AccumulatorFindLeaderType data;
        data.modelVersion = h->modelVersion;
        data.name = myName;
        findLeaderReduce = allReduceService->allReduce(
            group, "Accumulator::findLeader", std::move(data),
            [&](AccumulatorFindLeaderType& a, AccumulatorFindLeaderType& b) {
              if (std::tie(a.modelVersion, a.name) < std::tie(b.modelVersion, b.name)) {
                std::swap(a, b);
              }
            },
            [findLeaderResult = this->findLeaderResult,
             this](AccumulatorFindLeaderType* v, [[maybe_unused]] rpc::Error* error) {
              if (v) {
                *findLeaderResult = [this, modelVersion = v->modelVersion,
                                     leader = std::string(v->name)]() mutable noexcept {
                  isFindingLeader = false;
                  if (syncLeader.empty()) {
                    log.verbose("Leader is %s\n", leader);
                  } else if (syncLeader == leader) {
                    log.verbose("Leader is still %s\n", leader);
                  } else {
                    log.verbose("Leader changed from %s to %s\n", syncLeader, leader);
                  }
                  syncLeader = std::move(leader);
                  {
                    std::lock_guard l(group->mutex);
                    members_ = group->members;
                  }
                  log.verbose("Group has %d members\n", members_.size());
                  if (modelVersion != h->modelVersion || !hasReceivedModel_) {
                    requestModel();
                  }
                };
              } else {
                log.error("findLeader failed with error %s\n", error->what());
                *findLeaderResult = [this]() mutable {
                  if (h->syncId == group->syncId) {
                    resync();
                  }
                };
              }
            });
      }
    }

    if (h->haveNewParameters) {
      if (!isWaitingForModel && h->modelVersion != h->newModelVersion) {
        h->haveNewParameters = false;
        log.debug(
            "ignoring unexpected new params due to modelVersion mismatch (got %d, expected %d)\n", h->newModelVersion,
            h->modelVersion);
      } else {
        log.debug("DEBUG new params\n");
        commitModelUpdate();
        isWaitingForModel = false;
      }
    } else if (isWaitingForModel && now - isWaitingForModelTimestamp >= std::chrono::seconds(10)) {
      log.debug("Timed out waiting for model, retrying\n");
      requestModel();
    } else if (
        !isWaitingForModel && connectedImpl() && syncLeader != myName &&
        now - lastReceivedModel >= std::chrono::minutes(2)) {
      log.verbose("EMERGENCY RESYNC TO GET MODEL UPDATE!\n");
      lastReceivedModel = now;
      resync();
    }
    if (h->haveNewBuffers) {
      commitBuffersUpdate();
    }

    if (!members_.empty()) {
      sendModelUpdates();
    }

    if (logit) {
      log.debug("DEBUG connected: %d\n", connectedImpl());
      log.debug("DEBUG isWaitingForModel: %d\n", isWaitingForModel);
      log.debug("DEBUG isFindingLeader: %d\n", isFindingLeader);
      log.debug("DEBUG isCopyingGradients: %d\n", isCopyingGradients);
      log.debug("DEBUG nextGradientReductionIndex: %d\n", nextGradientReductionIndex);
      log.debug("DEBUG nextGradientReductionResultIndex: %d\n", nextGradientReductionResultIndex);
    }
  }

  py::object deepCopy(const py::handle& v) {
    if (v.ptr() == Py_True) {
      return py::reinterpret_borrow<py::object>(v);
    } else if (v.ptr() == Py_False) {
      return py::reinterpret_borrow<py::object>(v);
    } else if (v.ptr() == Py_None) {
      return py::reinterpret_borrow<py::object>(v);
    } else if (py::isinstance<py::float_>(v)) {
      return py::reinterpret_borrow<py::object>(v);
    } else if (py::isinstance<py::dict>(v)) {
      const py::dict& olddict = py::reinterpret_borrow<py::dict>(v);
      py::dict newdict;
      for (auto& [key, value] : olddict) {
        newdict[deepCopy(key)] = deepCopy(value);
      }
      return py::reinterpret_steal<py::object>(newdict.release());
    } else if (py::isinstance<py::str>(v)) {
      return py::reinterpret_borrow<py::object>(v);
    } else if (py::isinstance<py::array>(v)) {
      return py::reinterpret_borrow<py::object>(v);
    } else if (py::isinstance<py::int_>(v)) {
      return py::reinterpret_borrow<py::object>(v);
    } else if (py::isinstance<py::list>(v)) {
      const py::list& oldlist = py::reinterpret_borrow<py::list>(v);
      size_t n = oldlist.size();
      py::list newlist(n);
      for (size_t i = 0; i != n; ++i) {
        newlist[i] = deepCopy(oldlist[i]);
      }
      return py::reinterpret_steal<py::list>(newlist.release());
    } else if (auto t = rpc::tryFromPython(v)) {
      return rpc::toPython(t->to(rpc::kCPU, false, true));
    } else if (py::isinstance<py::tuple>(v)) {
      const py::tuple& oldtuple = py::reinterpret_borrow<py::tuple>(v);
      size_t n = oldtuple.size();
      py::tuple newtuple(n);
      for (size_t i = 0; i != n; ++i) {
        newtuple[i] = deepCopy(oldtuple[i]);
      }
      return py::reinterpret_steal<py::tuple>(newtuple.release());
    } else {
      return py::reinterpret_borrow<py::object>(v);
    }
  }

  py::object state() {
    glock l(h->mutex);
    hasNewUserState_ = false;
    return *userState_;
  }

  void setState(py::object userState) {
    userState = deepCopy(userState);
    glock l(h->mutex);
    userState_ = userState;
    wantsUserState_ = false;
    std::vector<rpc::Tensor> sendParameters;
    std::vector<rpc::Tensor> sendBuffers;

    for (auto& v : h->modelParameters) {
      sendParameters.push_back(v.to(rpc::kCPU, false, true));
    }
    for (auto& v : h->modelBuffers) {
      sendBuffers.push_back(v.to(rpc::kCPU, false, true));
    }

    auto now = std::chrono::steady_clock::now();

    if (isItTimeForRegularModelUpdateYet(now)) {
      lastSentModel = now;
      log.debug("Sending regular model/state update\n");
      for (auto& n : members_) {
        if (n != myName) {
          call<bool>(
              n, "AccumulatorService::modelUpdate", resName_, h->syncId, true, h->modelVersion, sendParameters,
              sendBuffers, userState_);
        }
      }
    } else {
      for (auto& n : h->requestedModelUpdate) {
        if (n != myName && std::find(members_.begin(), members_.end(), n) != members_.end()) {
          call<bool>(
              n, "AccumulatorService::modelUpdate", resName_, h->syncId, false, h->modelVersion, sendParameters,
              sendBuffers, userState_);
        }
      }
    }

    h->requestedModelUpdate.clear();

    log.verbose("got user state yey\n");
  }

  bool isItTimeForRegularModelUpdateYet(std::chrono::steady_clock::time_point now) {
    if (syncLeader == myName) {
      if (now - lastSentModel >= std::chrono::seconds(30)) {
        return true;
      }
    }
    return false;
  }

  void sendModelUpdates() {
    if (syncLeader != myName) {
      wantsUserState_ = false;
      h->requestedModelUpdate.clear();
      return;
    }
    auto now = std::chrono::steady_clock::now();
    if (!h->requestedModelUpdate.empty() || isItTimeForRegularModelUpdateYet(now)) {
      log.verbose("wants user state, requestedModelUpdate.size() is %d\n", h->requestedModelUpdate.size());
      wantsUserState_ = true;
    }
    if (syncLeader == myName) {
      if (now - lastSentBuffers >= std::chrono::seconds(12)) {
        lastSentBuffers = now;
        std::vector<rpc::Tensor> sendBuffers;
        for (auto& v : h->modelBuffers) {
          sendBuffers.push_back(v.to(rpc::kCPU, true, true));
        }
        for (auto& n : members_) {
          if (n != myName) {
            call<bool>(n, "AccumulatorService::buffersUpdate", resName_, h->syncId, sendBuffers);
          }
        }
      }
    }
  }

  void commitBuffersUpdate() {
    log.debug("Got new buffers, yey\n");
    h->haveNewBuffers = false;

    if (h->modelBuffers.size() != h->newBuffers.size()) {
      throw std::runtime_error("Model buffers size mismatch in update!");
    }

    for (size_t i = 0; i != h->modelBuffers.size(); ++i) {
      h->modelBuffers[i].copy_(h->newBuffers[i], true);
    }
  }

  void commitModelUpdate() {
    log.debug("Got new parameters, modelVersion %d -> %d\n", h->modelVersion, h->newModelVersion);
    h->haveNewParameters = false;
    h->haveNewBuffers = false;
    h->modelVersion = h->newModelVersion;

    lastReceivedModel = std::chrono::steady_clock::now();

    if (h->modelParameters.size() != h->newParameters.size()) {
      throw std::runtime_error("Model parameters size mismatch in update!");
    }
    if (h->modelBuffers.size() != h->newBuffers.size()) {
      throw std::runtime_error("Model parameters size mismatch in update!");
    }

    for (size_t i = 0; i != h->modelParameters.size(); ++i) {
      h->modelParameters[i].copy_(h->newParameters[i], true);
    }
    for (size_t i = 0; i != h->modelBuffers.size(); ++i) {
      h->modelBuffers[i].copy_(h->newBuffers[i], true);
    }

    py::gil_scoped_acquire gil;
    userState_ = std::move(h->newUserState);
    hasNewUserState_ = true;
    hasReceivedModel_ = true;
  }

  template<typename T, typename... Args>
  Future<T> call(std::string_view peerName, std::string_view funcName, Args&&... args) {
    return callImpl<T>(*rpc, peerName, funcName, std::forward<Args>(args)...);
  }

  std::mutex gradientsFreeListMutex;
  std::vector<std::vector<rpc::Tensor>> gradientsFreeList;
  std::vector<std::shared_ptr<ReduceGradientsContainer>> gradientReductions;

  std::vector<rpc::Tensor> allocateGradients() {
    std::lock_guard l(gradientsFreeListMutex);
    if (!gradientsFreeList.empty()) {
      auto r = std::move(gradientsFreeList.back());
      gradientsFreeList.pop_back();
      return r;
    }

    bool isCuda = gradsOnCuda;

    std::vector<rpc::Tensor> r;

    for (rpc::Tensor& v : h->modelParameters) {
      rpc::Tensor grad = v.mutable_grad();
      if (v.requires_grad()) {
        if (!grad.defined()) {
          grad = rpc::zeros_like(v);
          v.set_grad(grad);
        }
        if (isCuda) {
          r.push_back(rpc::zeros_like(grad, rpc::kCPU).pin_memory());
        } else {
          r.push_back(rpc::zeros_like(grad, rpc::kCPU));
        }
      }
    }
    return r;
  }
  void freeGradients(std::vector<rpc::Tensor>&& grads) {
    std::lock_guard l(gradientsFreeListMutex);
    gradientsFreeList.push_back(std::move(grads));
  }

  void reduceImpl(int batchSize) {
    if (!wantsGradients()) {
      fatal("reduceGradients/skipGradients called while wantsGradients() is false");
    }

    log.debug("Reduce %d\n", batchSize);

    glock l(h->mutex);

    size_t index = nextGradientReductionIndex;
    std::shared_ptr<ReduceGradientsContainer> target = gradientReductions[index];
    if (target && target->reduceStarted && !target->reduceDone) {
      fatal("reduceImpl internal error, reduce already started!");
    }
    if (!target || target->reduceDone) {
      target = gradientReductions[index] = std::make_shared<ReduceGradientsContainer>(this);
    }

    if (nextGradientReductionIndex == gradientReductions.size() - 1) {
      nextGradientReductionIndex = 0;
    } else {
      ++nextGradientReductionIndex;
    }

    isCopyingGradients = true;

    std::optional<rpc::CUDAStream> stream;
    if (gradsOnCuda) {
      stream.emplace(rpc::getCurrentCUDAStream());
    }

    target->syncId = h->syncId;

    // async.run([this, batchSize, target, index, stream = std::move(stream), syncId = h->syncId]() mutable noexcept {
    Dtor dtor = [&] {
      // std::lock_guard l(h->mutex);
      actuallyZeroGradients();
      isCopyingGradients = false;
    };
    std::optional<rpc::CUDAStreamGuard> sg;
    if (stream) {
      sg.emplace(*stream);
    }
    rpc::AutoGradMode ng(false);
    bool synchronize = false;
    if (batchSize) {
      ++target->data.numGradients;
      target->data.batchSize += batchSize;
      bool add = true;
      if (target->data.gradients.empty()) {
        target->data.gradients = allocateGradients();
        add = false;
      }
      auto& targetGradients = target->data.gradients;
      size_t i = 0;
      std::vector<rpc::Tensor> addGrads;
      for (auto& v : h->modelParameters) {
        auto grad = v.grad();
        if (grad.defined()) {
          if (i == targetGradients.size()) {
            fatal("grads grew?");
          }
          if (add) {
            addGrads.push_back(grad.to(targetGradients[i].device(), true));
          } else {
            targetGradients[i].copy_(grad, true);
          }
          ++i;
        }
      }
      if (i != targetGradients.size()) {
        fatal("grads shrank?");
      }
      synchronize = gradsOnCuda;
      if (synchronize && stream) {
        stream->synchronize();
      }
      if (add) {
        for (size_t i = 0; i != addGrads.size(); ++i) {
          targetGradients[i].add_(addGrads[i]);
        }
        if (synchronize && stream) {
          stream->synchronize();
        }
      }
    } else {
      ++target->data.numSkipped;
    }
    if (synchronize && stream) {
      stream->synchronize();
    }

    try {
      // std::lock_guard l(h->mutex);
      if (target->syncId == h->syncId && target->syncId == group->syncId) {
        if (target->isCounting) {
          log.debug("Already counting!\n");
          target->wantsMoreCounting = true;
        } else {
          log.debug("Start new count!\n");
          startCount(index, std::move(target));
        }
      }
    } catch (const std::exception& e) {
      fatal("exception %s\n", e.what());
    }
    //});
  }

  void startCount(size_t index, std::shared_ptr<ReduceGradientsContainer> target) {
    if (target->syncId != h->syncId || target->syncId != group->syncId) {
      return;
    }
    log.debug("startCount index %d called with local batch size of %d\n", index, target->data.batchSize);
    target->isCounting = true;
    target->reduce = allReduceService->allReduce(
        group, fmt::sprintf("Accumulator reduce size %d", index), target->data.batchSize,
        [&](size_t& a, size_t& b) { a += b; },
        [target, this, index](size_t* size, [[maybe_unused]] rpc::Error* error) {
          auto& result = target->result;
          if (size) {
            log.verbose("reduce ready batch size %d/%d\n", *size, virtualBatchSize);
            if (*size >= virtualBatchSize) {
              log.debug("That's enough, start reduce!\n");
              result = [target = std::move(target), this, index]() mutable {
                if (target->syncId != h->syncId || target->syncId != group->syncId) {
                  return;
                }
                target->reduceStarted = true;
                target->reduce = allReduceService->allReduce(
                    group, fmt::sprintf("Accumulator reduce %d", index), target->data,
                    [&](ReductionType& a, ReductionType& b) { a.add(b); },
                    [target, this](ReductionType* v, [[maybe_unused]] rpc::Error* error) {
                      if (v) {
                        auto& result = target->result;
                        result = [target = std::move(target), this, result = std::move(*v)]() mutable {
                          target->reduceDone = true;
                          log.debug(
                              "reduce done, batch size %d/%d  (%d grads, %d skipped)\n", result.batchSize,
                              virtualBatchSize, result.numGradients, result.numSkipped);
                          setGradients(result);
                        };
                      } else {
                        log.error("Accumulator reduction failed; %s", error->what());
                        target->result = [this]() mutable { onError(); };
                      }
                    });
              };
            } else {
              log.debug("Not enough, start new count!\n");
              result = [target = std::move(target), this, index]() mutable {
                target->isCounting = false;
                if (target->wantsMoreCounting) {
                  startCount(index, target);
                }
              };
            }
          } else {
            log.error("Accumulator reduction failed; %s", error->what());
            result = [this]() mutable { onError(); };
          }
        });
  }

  void skipGradients() {
    reduceImpl(0);
  }

  float sum(std::vector<rpc::Tensor> grads) {
    rpc::Tensor sum;
    for (auto& v : grads) {
      if (sum.defined()) {
        sum += v.sum();
      } else {
        sum = v.sum();
      }
    }
    return sum.item<float>();
  }

  void reduceGradients(int batchSize) {
    reduceImpl(batchSize);
  }
};

Accumulator::Accumulator(std::string name, pybind11::object parameters, pybind11::object buffers, const Group* group) {
  if (group) {
    impl = std::make_unique<AccumulatorImpl>(*group, name, parameters, buffers);
  } else {
    impl = std::make_unique<AccumulatorImpl>(name, parameters, buffers);
  }
}

Accumulator::~Accumulator() {}

void Accumulator::update() {
  impl->update();
}

void Accumulator::connect(std::string address) {
  impl->connect(address);
}

bool Accumulator::connected() {
  return impl->connected();
}

bool Accumulator::wantsState() {
  return impl->wantsState();
}

bool Accumulator::hasNewState() {
  return impl->hasNewState();
}

bool Accumulator::hasGradients() {
  return impl->hasGradients();
}

bool Accumulator::wantsGradients() {
  return impl->wantsGradients();
}

void Accumulator::setState(pybind11::object userState) {
  impl->setState(std::move(userState));
}

pybind11::object Accumulator::state() {
  return impl->state();
}

void Accumulator::skipGradients() {
  impl->skipGradients();
}

void Accumulator::reduceGradients(int batchSize) {
  impl->reduceGradients(batchSize);
}

void Accumulator::zeroGradients() {
  impl->zeroGradients();
}

py::dict Accumulator::getGradientStats() const {
  py::dict r;
  auto stats = impl->getGradientStats();
  r["num_gradients"] = stats.numGradients;
  r["num_skipped"] = stats.numSkipped;
  r["batch_size"] = stats.batchSize;
  return r;
}

int64_t Accumulator::modelVersion() const {
  return impl->h->modelVersion;
}

void Accumulator::setModelVersion(int64_t version) {
  impl->h->modelVersion = version;
}

void Accumulator::setVirtualBatchSize(int n) {
  impl->setVirtualBatchSize(n);
}

void Accumulator::setParallelGradients(int n) {
  impl->setParallelGradients(n);
}

std::string Accumulator::getLeader() {
  return impl->getLeader();
}

bool Accumulator::isLeader() {
  return impl->isLeader();
}

} // namespace moolib
