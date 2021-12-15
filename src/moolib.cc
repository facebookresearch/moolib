/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "accumulator.h"
#include "batch_utils.h"
#include "broker.h"
#include "env.h"
#include "group.h"
#include "logging.h"
#include "pythonserialization.h"
#include "rpc.h"
#include "shm.h"
#include "synchronization.h"
#include "util.h"

#include "pybind11/functional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <list>
#include <vector>

namespace rpc {
void setPythonTensor(pybind11::handle o, const rpc::Tensor& t);
}

namespace moolib {

std::once_flag importThreadingFlag;
void importThreading() {
  // threading module must be imported, since we use threads.
  // Otherwise, there may be errors.
  py::module::import("threading");
}

namespace {

struct PyThreadKeepAlive {
  PyThreadState* tstate = nullptr;
  PyThreadKeepAlive() = default;
  PyThreadKeepAlive(PyThreadState* tstate) : tstate(tstate) {
    ++tstate->gilstate_counter;
  }
  PyThreadKeepAlive(const PyThreadKeepAlive&) = delete;
  PyThreadKeepAlive(PyThreadKeepAlive&& n) {
    tstate = std::exchange(n.tstate, nullptr);
  }
  ~PyThreadKeepAlive() {
    if (tstate && --tstate->gilstate_counter == 0) {
      PyThreadState_Clear(tstate);
      if (!_Py_IsFinalizing()) {
        PyThreadState_Delete(tstate);
      }
    }
  }
  PyThreadKeepAlive& operator=(const PyThreadKeepAlive&) = delete;
  PyThreadKeepAlive& operator=(PyThreadKeepAlive&& n) {
    std::swap(tstate, n.tstate);
    return *this;
  }
};

struct ModuleState {
  std::unordered_map<PyThreadState*, PyThreadKeepAlive> keepalive;
  py::object get_running_loop;
};

ModuleState* moduleState = nullptr;

// Keep python thread state alive for as long as this module exists.
// This prevents pybind from creating and destroying thread states
// over and over in callbacks.
void keepThreadAlive() {
  if (!moduleState) {
    fatal("moolib module is not loaded");
  }
  PyThreadState* tstate = _PyThreadState_UncheckedGet();
  auto i = moduleState->keepalive.find(tstate);
  if (i != moduleState->keepalive.end()) {
    return;
  }
  moduleState->keepalive[tstate] = tstate;
}

int moduleRefcount = 0;

void moduleExit();

void moduleIncRef() {
  if (++moduleRefcount == 1) {
    moduleState = new ModuleState();
  }
}

void moduleDecRef() {
  if (--moduleRefcount == 0) {
    delete moduleState;
    moduleState = nullptr;
  }
}

void moduleUnload() {
  moduleDecRef();
  py::gil_scoped_release gil;
  moduleExit();
}

std::atomic_int rpcLiveCount = 0;
std::mutex rpcListMutex;
std::list<std::weak_ptr<struct RpcCounted>> rpcList;

struct RpcCounted : rpc::Rpc {
  std::optional<decltype(rpcList)::iterator> rpcListIterator;
  std::weak_ptr<RpcCounted> weakPtr;
  void init(std::weak_ptr<RpcCounted> weakPtr) {
    this->weakPtr = weakPtr;
    rpcLiveCount.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard l(rpcListMutex);
    rpcListIterator = rpcList.insert(rpcList.end(), weakPtr);
  }
  ~RpcCounted() {
    std::lock_guard l(rpcListMutex);
    if (rpcListIterator) {
      rpcList.erase(*rpcListIterator);
      rpcLiveCount.fetch_sub(1, std::memory_order_relaxed);
    }
  }
};

std::atomic_bool isExiting = false;

void moduleExit() {
  if (isExiting.exchange(true)) {
    fatal("internal error: moduleExit called twice concurrently");
  }
  std::unique_lock l(rpcListMutex);
  size_t n = rpcLiveCount;
  if (rpcList.size() != n) {
    fatal("internal error: rpcList.size() (%d) != rpcLiveCount (%d)", rpcList.size(), n);
  }
  if (!rpcList.empty()) {
    log.info("Note: cleaning up %d leaked Rpc objects on exit", n);
    while (!rpcList.empty()) {
      auto toRemove = std::move(rpcList);
      for (auto& i : toRemove) {
        if (auto p = i.lock()) {
          p->rpcListIterator.reset();
          rpcLiveCount.fetch_sub(1, std::memory_order_relaxed);
        } else {
          fatal("internal error: lock failed");
        }
      }
      rpcList.clear();
      l.unlock();
      for (auto& i : toRemove) {
        if (auto p = i.lock()) {
          p->close();
        }
      }
      l.lock();
    }
  }
  isExiting = false;
}

} // namespace

struct TimeoutError : std::exception {
  TimeoutError() = default;
  const char* what() const noexcept override final {
    return "The operation timed out";
  }
};

struct CancelledError : std::exception {
  CancelledError() = default;
  const char* what() const noexcept override final {
    return "The operation was cancelled";
  }
};

struct FutureWrapper {

  GilWrapper<py::object> value;
  std::atomic_int flags = 0;
  rpc::Semaphore sem;
  std::atomic<rpc::FunctionPointer> callback = nullptr;

  rpc::SpinMutex errorMutex;
  std::optional<rpc::Error> error;

  std::weak_ptr<FutureWrapper> weakPtr;

  // Cannot inherit from std::shared_from_this, due to pybind bugs
  std::shared_ptr<FutureWrapper> shared() {
    return std::shared_ptr<FutureWrapper>(weakPtr);
  }

  ~FutureWrapper() {
    if (callback.load(std::memory_order_relaxed)) {
      fatal("internal error: ~FutureWrapper has non-null callback");
    }
  }

  bool done() {
    return flags != 0;
  }

  void doCallback() {
    if (callback.load(std::memory_order_relaxed)) {
      rpc::FunctionPointer f = callback.exchange(nullptr);
      if (f) {
        (rpc::Function<void()>(f))();
      }
    }
  }

  void setResult(GilWrapper<py::object> result) {
    value = std::move(result);
    flags |= 1;
    sem.post();
    doCallback();
  }
  template<typename T>
  void setException(T&& error) {
    {
      std::lock_guard l(errorMutex);
      this->error = std::forward<T>(error);
    }
    flags |= 2;
    sem.post();
    doCallback();
  }
  void cancel() {
    flags |= 4;
    sem.post();
    doCallback();
  }

  py::object get() {
    int f = flags.load(std::memory_order_relaxed);
    if (f & 1) {
      if (!value) {
        throw std::runtime_error("Future value not available. Maybe it has already been retrieved");
      }
      return value.release();
    } else if (f & 2) {
      std::lock_guard l(errorMutex);
      throw *error;
    } else if (f & 4) {
      throw CancelledError();
    } else {
      throw std::runtime_error("Future::get() called in invalid state");
    }
  }
  py::object result() {
    if (!done()) {
      wait();
    }
    return get();
  }
  py::object result(float timeout) {
    if (!done()) {
      wait(timeout);
    }
    return get();
  }
  py::object exception() {
    int f = flags.load(std::memory_order_relaxed);
    if (f & 2) {
      std::lock_guard l(errorMutex);
      return py::cast(*error);
    } else {
      return py::none();
    }
  }
  void wait() {
    py::gil_scoped_release gil;
    while (!done()) {
      sem.wait();
    }
  }
  void wait(float timeout) {
    py::gil_scoped_release gil;
    auto now = std::chrono::steady_clock::now();
    auto end = now + std::chrono::duration<float, std::ratio<1, 1>>(timeout);
    while (!done()) {
      if (now >= end) {
        throw TimeoutError();
      }
      sem.wait_until(end);
      now = std::chrono::steady_clock::now();
    }
  }

  py::object await() {
    if (!moduleState) {
      fatal("moolib module is not loaded");
    }
    if (!moduleState->get_running_loop) {
      moduleState->get_running_loop = py::module::import("asyncio").attr("get_running_loop");
    }
    py::object loop = moduleState->get_running_loop();
    py::object future = loop.attr("create_future")();

    callback = rpc::Function<void()>([me = shared(), loop = GilWrapper<py::object>(std::move(loop)),
                                      future = GilWrapper<py::object>(future)]() mutable noexcept {
                 int f = me->flags.load(std::memory_order_relaxed);
                 py::gil_scoped_acquire gil;
                 if (_Py_IsFinalizing()) {
                   return;
                 }
                 try {
                   keepThreadAlive();
                   if (f & 1) {
                     loop->attr("call_soon_threadsafe")(
                         py::cpp_function([me = std::move(me), future = std::move(future)]() mutable {
                           try {
                             if (!me->value) {
                               future->attr("set_exception")(py::reinterpret_borrow<py::object>(PyExc_RuntimeError)(
                                   "Future value not available. Maybe it has already been retrieved"));
                             } else {
                               future->attr("set_result")(me->value.release());
                             }
                           } catch (const py::error_already_set& e) {
                             // InvalidStateError probably means that the future was cancelled
                             if (!e.matches(py::module::import("asyncio").attr("InvalidStateError"))) {
                               throw;
                             }
                           }
                         }));
                   } else if (f & 2) {
                     try {
                       loop->attr("call_soon_threadsafe")(py::cpp_function([me, future = std::move(future)]() mutable {
                         try {
                           std::lock_guard l(me->errorMutex);
                           future->attr("set_exception")(
                               py::reinterpret_borrow<py::object>(PyExc_RuntimeError)(me->error->what()));
                         } catch (const py::error_already_set& e) {
                           if (!e.matches(py::module::import("asyncio").attr("InvalidStateError"))) {
                             throw;
                           }
                         }
                       }));
                     } catch (const py::error_already_set& e) {
                       std::string s;
                       {
                         std::lock_guard l(me->errorMutex);
                         s = me->error->what();
                       }
                       fatal("Python exception during callback: %s\nOriginal exception: %s", e.what(), s);
                     }
                   } else if (f & 4) {
                     loop->attr("call_soon_threadsafe")(
                         py::cpp_function([future = std::move(future)]() mutable { future->attr("cancel")(); }));
                   } else {
                     fatal("Future callback called in invalid state");
                   }
                 } catch (const py::error_already_set& e) {
                   fatal("Python exception during callback: %s", e.what());
                 }
                 future.reset();
                 loop.reset();
               }).release();
    if (flags.load() != 0) {
      doCallback();
    }
    return future.attr("__await__")();
  }
};

template<typename Future>
struct Promise {
  std::shared_ptr<Future> future;
  Promise() {
    future = std::make_shared<Future>();
    future->weakPtr = future;
  }
  Promise(const Promise&) = delete;
  Promise(Promise&&) = default;
  ~Promise() {
    if (future && !future->done()) {
      future->cancel();
    }
  }

  void setResult(GilWrapper<py::object> value) {
    future->setResult(std::move(value));
  }
  template<typename T>
  void setException(T&& error) {
    future->setException(std::forward<T>(error));
  }
  void cancel() {
    future->cancel();
  }

  std::shared_ptr<Future>& getFuture() {
    return future;
  }
};

struct QueueEntry {
  rpc::RpcDeferredReturn<GilWrapper<py::object>> ret;
  std::optional<GilWrapper<py::args>> args;
  std::optional<GilWrapper<py::kwargs>> kwargs;
  std::chrono::steady_clock::time_point timestamp;
};

struct QueueWrapper {
  int64_t batchSize = 0;
  int64_t batchDim = 0;

  std::chrono::milliseconds timeout;
  rpc::SpinMutex mutex;
  std::deque<QueueEntry> queue;
  rpc::Semaphore sem;
  std::deque<Promise<FutureWrapper>> waiters;

  void setResult(
      Promise<FutureWrapper>& promise, rpc::RpcDeferredReturn<GilWrapper<py::object>>&& ret,
      std::optional<GilWrapper<py::args>>&& args, std::optional<GilWrapper<py::kwargs>>&& kwargs) {
    py::object o;
    {
      py::gil_scoped_acquire gil;
      if (_Py_IsFinalizing()) {
        return;
      }
      keepThreadAlive();
      py::tuple tup(3);
      tup[0] = py::cast(std::move(ret), py::return_value_policy::move);
      tup[1] = args ? (py::object)*std::move(*args) : (py::object)py::none();
      tup[2] = kwargs ? (py::object)*std::move(*kwargs) : (py::object)py::none();
      o = std::move(tup);
    }
    promise.setResult(std::move(o));
  }

  void setBatchResult(
      std::vector<std::tuple<
          rpc::RpcDeferredReturn<GilWrapper<py::object>>, std::optional<GilWrapper<py::args>>,
          std::optional<GilWrapper<py::kwargs>>>>&& batch,
      Promise<FutureWrapper>& result) const {
    py::object o;
    {
      py::gil_scoped_acquire gil;
      if (_Py_IsFinalizing()) {
        return;
      }
      keepThreadAlive();
      const int64_t curBatchSize = batch.size();
      // Somehow std::vector<py::handle> doesn't work here.
      py::tuple src(curBatchSize);
      std::vector<rpc::RpcDeferredReturn<GilWrapper<py::object>>> retCallbacks;
      retCallbacks.reserve(curBatchSize);
      for (int64_t i = 0; i < curBatchSize; ++i) {
        auto& [ret, args, kwargs] = batch[i];
        retCallbacks.push_back(std::move(ret));
        src[i] = py::make_tuple(
            args ? (py::object)*std::move(*args) : (py::object)py::none(),
            kwargs ? (py::object)*std::move(*kwargs) : (py::object)py::none());
      }

      rpc::RpcDeferredReturn<GilWrapper<py::object>> retCallback;
      retCallback.f = [curBatchSize, dim = batchDim,
                       retCallbacks = std::move(retCallbacks)](GilWrapper<py::object> input) mutable {
        py::tuple batch = utils::unstackFields(*input, curBatchSize, dim);
        for (int64_t i = 0; i < curBatchSize; ++i) {
          py::object cur = std::move(batch[i]);
          retCallbacks[i](std::move(cur));
        }
      };
      py::tuple args = py::reinterpret_borrow<py::tuple>(utils::stackFields(src, batchDim));
      o = py::make_tuple(
          py::cast(std::move(retCallback), py::return_value_policy::move), std::move(args[0]), std::move(args[1]));
    }
    result.setResult(std::move(o));
  }

  void enqueue(
      rpc::RpcDeferredReturn<GilWrapper<py::object>> ret, std::optional<GilWrapper<py::args>> args,
      std::optional<GilWrapper<py::kwargs>> kwargs) {
    {
      std::unique_lock l(mutex);
      auto timeout = this->timeout;
      auto now = std::chrono::steady_clock::now();
      while (!queue.empty() && now >= queue.front().timestamp + timeout) {
        queue.pop_front();
      }
      if (!waiters.empty()) {
        auto promise = std::move(waiters.front());
        waiters.pop_front();
        l.unlock();
        if (batchSize > 0) {
          std::vector<std::tuple<
              rpc::RpcDeferredReturn<GilWrapper<py::object>>, std::optional<GilWrapper<py::args>>,
              std::optional<GilWrapper<py::kwargs>>>>
              batch = {std::make_tuple(std::move(ret), std::move(args), std::move(kwargs))};
          setBatchResult(std::move(batch), promise);
        } else {
          setResult(promise, std::move(ret), std::move(args), std::move(kwargs));
        }
        return;
      }
      queue.emplace_back();
      auto& e = queue.back();
      e.ret = std::move(ret);
      e.args = std::move(args);
      e.kwargs = std::move(kwargs);
      e.timestamp = now;
    }
    sem.post();
  }

  py::object await() {
    Promise<FutureWrapper> promise;
    py::object r = promise.getFuture()->await();
    {
      std::unique_lock l(mutex);
      if (!queue.empty()) {
        if (batchSize > 0) {
          const int64_t queSize = queue.size();
          const int64_t curBatchSize = std::min(batchSize, queSize);
          std::vector<std::tuple<
              rpc::RpcDeferredReturn<GilWrapper<py::object>>, std::optional<GilWrapper<py::args>>,
              std::optional<GilWrapper<py::kwargs>>>>
              batch;
          batch.reserve(curBatchSize);
          for (int64_t i = 0; i < curBatchSize; ++i) {
            auto e = std::move(queue.front());
            batch.emplace_back(std::move(e.ret), std::move(e.args), std::move(e.kwargs));
            queue.pop_front();
          }
          l.unlock();
          setBatchResult(std::move(batch), promise);
        } else {
          auto e = std::move(queue.front());
          queue.pop_front();
          l.unlock();
          setResult(promise, std::move(e.ret), std::move(e.args), std::move(e.kwargs));
        }
      } else {
        waiters.push_back(std::move(promise));
      }
    }
    return r;
  }
};

template<typename T>
struct DontCopy {
  T value;
  DontCopy() = default;
  DontCopy(const DontCopy&) {}
  DontCopy(DontCopy&&) {}
  DontCopy& operator=(const DontCopy&) {
    return *this;
  }
  DontCopy& operator=(DontCopy&&) {
    return *this;
  }
  T& operator*() {
    return value;
  }
};

template<typename MetaType>
struct Batcher {
  GilWrapper<py::object> target;
  int64_t nextStackIndex = 0;
  int64_t batchSize = 0;
  int64_t batchDimension = 0;
  rpc::Device device{"cpu"};
  int nTensors = 0;
  int currentTensor = 0;
  int64_t catBatchInputOffset = 0;
  int64_t catBatchInputSize = 0;
  int64_t catBatchOutputOffset = 0;
  bool isDoingCat = false;
  std::vector<int64_t> sizes;
  std::vector<MetaType> metaData;
  DontCopy<std::mutex> batchMutex;
  DontCopy<std::mutex> unbatchMutex;

  Batcher() = default;
  Batcher(int64_t batchSize) : batchSize(batchSize) {}
  Batcher(int64_t batchSize, std::string device, int64_t batchDimension)
      : batchSize(batchSize), batchDimension(batchDimension), device(device) {}

  template<bool cat>
  py::object prepareForBatchCopy(const py::handle& v) {
    if (py::isinstance<py::dict>(v)) {
      const py::dict& dict = py::reinterpret_borrow<py::dict>(v);
      py::dict newdict;
      for (auto& [key, value] : dict) {
        newdict[key] = prepareForBatchCopy<cat>(value);
      }
      return std::move(newdict);
    } else if (py::isinstance<py::list>(v)) {
      const py::list& list = py::reinterpret_borrow<py::list>(v);
      size_t n = list.size();
      py::list newlist(n);
      for (size_t i = 0; i != n; ++i) {
        newlist[i] = prepareForBatchCopy<cat>(list[i]);
      }
      return std::move(newlist);
    } else if (auto t = rpc::tryFromPython(v)) {
      auto s = t->sizes();
      if ((int64_t)s.size() <= (cat ? batchDimension : batchDimension - 1)) {
        throw std::runtime_error(fmt::sprintf(
            "Given input tensor with %d dimensions, cannot %s in dimension %d", s.size(), cat ? "cat" : "stack",
            batchDimension));
      }
      if (cat) {
        sizes.assign(s.begin(), s.end());
        sizes[batchDimension] = batchSize;
      } else {
        sizes.resize(1 + s.size());
        std::copy(s.begin(), s.begin() + batchDimension, sizes.begin());
        std::copy(s.begin() + batchDimension, s.end(), sizes.begin() + batchDimension + 1);
        sizes[batchDimension] = batchSize;
      }
      rpc::Tensor tensor = rpc::empty(rpc::IntArrayRef(sizes.data(), sizes.size()), t->scalar_type(), device);
      if (cat) {
        int64_t offset = catBatchInputOffset;
        int64_t n = s[batchDimension];
        if (offset > n) {
          fatal("Batch internal error: offset > n");
        }
        if (nTensors == 0) {
          catBatchInputSize = n;
        } else {
          if (n != catBatchInputSize) {
            throw std::runtime_error(fmt::sprintf(
                "Batch dimension size mismatch; during a cat operation, all tensors must have the same size in the "
                "batch dimension (%d). Got %d and %d",
                batchDimension, catBatchInputSize, n));
          }
        }
        n -= offset;
        if (n <= batchSize && offset == 0) {
          tensor.narrow(batchDimension, 0, n).copy_(*t);
        } else {
          n = std::min(n, batchSize);
          tensor.narrow(batchDimension, 0, n).copy_(t->narrow(batchDimension, offset, n));
        }
      } else {
        tensor.select(batchDimension, 0).copy_(*t);
      }
      ++nTensors;
      return rpc::toPython(tensor);
    } else if (py::isinstance<py::tuple>(v)) {
      const py::tuple& tuple = py::reinterpret_borrow<py::tuple>(v);
      size_t n = tuple.size();
      py::tuple newtuple(n);
      for (size_t i = 0; i != n; ++i) {
        newtuple[i] = prepareForBatchCopy<cat>(tuple[i]);
      }
      return std::move(newtuple);
    } else {
      return py::reinterpret_borrow<py::object>(v);
    }
  }

  template<bool cat>
  void visit(const py::handle& dest, const py::handle& source) {
    if (py::isinstance<py::dict>(dest)) {
      if (!py::isinstance<py::dict>(source)) {
        throw std::runtime_error("type mismatch in batch operation");
      }
      const py::dict& sourceDict = py::reinterpret_borrow<py::dict>(source);
      const py::dict& destDict = py::reinterpret_borrow<py::dict>(dest);
      for (auto& [key, value] : destDict) {
        visit<cat>(value, sourceDict[key]);
      }
    } else if (py::isinstance<py::list>(dest)) {
      if (!py::isinstance<py::list>(source)) {
        throw std::runtime_error("type mismatch in batch operation");
      }
      const py::list& sourceList = py::reinterpret_borrow<py::list>(source);
      const py::list& destList = py::reinterpret_borrow<py::list>(dest);
      size_t n = destList.size();
      for (size_t i = 0; i != n; ++i) {
        visit<cat>(destList[i], sourceList[i]);
      }
    } else if (auto destT = rpc::tryFromPython(dest)) {
      auto sourceT = rpc::tryFromPython(source);
      if (!sourceT) {
        throw std::runtime_error("type mismatch in batch operation");
      }
      auto s = sourceT->sizes();
      if ((int64_t)s.size() <= (cat ? batchDimension : batchDimension - 1)) {
        throw std::runtime_error(fmt::sprintf(
            "Given input tensor with %d dimensions, cannot %s in dimension %d", s.size(), cat ? "cat" : "stack",
            batchDimension));
      }
      if (cat) {
        int64_t inputOffset = catBatchInputOffset;
        int64_t n = s[batchDimension];
        if (inputOffset > n) {
          fatal("Batch internal error: offset > n");
        }
        if (currentTensor == 0) {
          catBatchInputSize = n;
        } else {
          if (n != catBatchInputSize) {
            throw std::runtime_error(fmt::sprintf(
                "Batch dimension size mismatch; during a cat operation, all tensors must have the same size in the "
                "batch dimension (%d). Got %d and %d",
                batchDimension, catBatchInputSize, n));
          }
        }
        int64_t outputOffset = catBatchOutputOffset;
        int64_t left = batchSize - outputOffset;
        n -= inputOffset;
        if (n <= left && inputOffset == 0) {
          destT->narrow(batchDimension, outputOffset, n).copy_(*sourceT);
        } else {
          n = std::min(n, left);
          destT->narrow(batchDimension, outputOffset, n).copy_(sourceT->narrow(batchDimension, inputOffset, n));
        }
      } else {
        destT->select(batchDimension, nextStackIndex).copy_(*sourceT);
      }
      ++currentTensor;
    } else if (py::isinstance<py::tuple>(dest)) {
      if (!py::isinstance<py::tuple>(source)) {
        throw std::runtime_error("type mismatch in batch operation");
      }
      const py::tuple& sourceTuple = py::reinterpret_borrow<py::tuple>(source);
      const py::tuple& destTuple = py::reinterpret_borrow<py::tuple>(dest);
      size_t n = destTuple.size();
      for (size_t i = 0; i != n; ++i) {
        visit<cat>(destTuple[i], sourceTuple[i]);
      }
    }
  }

  template<typename Callback>
  void cat(py::object value, Callback&& callback) {
    int64_t localInputOffset = 0;
    while (true) {
      std::unique_lock l(*batchMutex);
      catBatchInputOffset = localInputOffset;
      catBatchInputSize = 0;
      if (!target) {
        catBatchOutputOffset = 0;
        nTensors = 0;
        target = prepareForBatchCopy<true>(value);
        if (!*target) {
          target = value;
        }
        isDoingCat = true;
      } else {
        if (!isDoingCat) {
          throw std::runtime_error(
              "Batcher.cat: Previously called with stack; cannot mix cat/stack within the same batch");
        }
        currentTensor = 0;
        visit<true>(*target, value);
        if (currentTensor != nTensors) {
          throw std::runtime_error(fmt::sprintf(
              "num tensors mismatch in batch operation; got %d tensors, batch has %d", currentTensor, nTensors));
        }
      }
      int64_t inputSize = catBatchInputSize - localInputOffset;
      int64_t left = batchSize - catBatchOutputOffset;
      if (inputSize >= left) {
        py::object r = std::move(*target);
        target.reset();
        l.unlock();
        callback(std::move(r));
        if (inputSize == left) {
          break;
        } else {
          localInputOffset += left;
        }
      } else {
        catBatchOutputOffset += inputSize;
        break;
      }
    }
  }

  std::optional<std::pair<py::object, std::vector<MetaType>>> stack(py::object value, MetaType meta) {
    std::lock_guard l(*batchMutex);
    if (!target) {
      nTensors = 0;
      target = prepareForBatchCopy<false>(value);
      if (!*target) {
        target = value;
      }
      metaData.resize(batchSize);
      metaData[0] = std::move(meta);
      nextStackIndex = 1;
      isDoingCat = false;
    } else {
      if (isDoingCat) {
        throw std::runtime_error(
            "Batcher.stack: Previously called with cat; cannot mix cat/stack within the same batch");
      }
      currentTensor = 0;
      visit<false>(*target, value);
      if (currentTensor != nTensors) {
        throw std::runtime_error(fmt::sprintf(
            "num tensors mismatch in batch operation; got %d tensors, batch has %d", currentTensor, nTensors));
      }
      metaData[nextStackIndex] = std::move(meta);
      ++nextStackIndex;
    }
    if (nextStackIndex == batchSize) {
      py::object r = std::move(*target);
      target.reset();
      return std::make_pair(std::move(r), std::move(metaData));
    }
    return {};
  }

  void prepareForUnbatch(const py::handle& v, std::vector<std::pair<py::object, rpc::Tensor>>& tensors) {
    if (py::isinstance<py::dict>(v)) {
      const py::dict& dict = py::reinterpret_borrow<py::dict>(v);
      for (auto& [key, value] : dict) {
        prepareForUnbatch(value, tensors);
      }
    } else if (py::isinstance<py::list>(v)) {
      const py::list& list = py::reinterpret_borrow<py::list>(v);
      size_t n = list.size();
      for (size_t i = 0; i != n; ++i) {
        prepareForUnbatch(list[i], tensors);
      }
    } else if (auto t = rpc::tryFromPython(v)) {
      tensors.emplace_back(py::reinterpret_borrow<py::object>(v), t->to(rpc::kCPU));
    } else if (py::isinstance<py::tuple>(v)) {
      const py::tuple& tuple = py::reinterpret_borrow<py::tuple>(v);
      size_t n = tuple.size();
      for (size_t i = 0; i != n; ++i) {
        prepareForUnbatch(tuple[i], tensors);
      }
    }
  }

  std::vector<std::pair<py::object, rpc::Tensor>> unbatchTensors;

  template<typename T>
  void unbatch(py::object v, T&& callbacks) {
    std::lock_guard l(*unbatchMutex);
    auto& tensors = unbatchTensors;
    prepareForUnbatch(v, tensors);
    size_t n = callbacks.size();
    for (size_t i = 0; i != n; ++i) {
      for (auto& [o, t] : tensors) {
        if (t.size(0) != (int64_t)n) {
          throw std::runtime_error(
              fmt::sprintf("unexpected batch size in output tensor: got %d, expected %d", t.size(0), n));
        }
        setPythonTensor(o, t[i]);
      }
      callbacks[i](v);
    }

    unbatchTensors.clear();
  }
};

namespace {

py::object getBatchValue(std::optional<GilWrapper<py::args>>& args, std::optional<GilWrapper<py::kwargs>>& kwargs) {
  py::object o;
  if (args) {
    if (kwargs) {
      py::tuple tup(2);
      tup[0] = *std::move(*args);
      tup[1] = *std::move(*kwargs);
      o = tup;
    } else {
      o = *std::move(*args);
    }
  } else if (kwargs) {
    o = *std::move(*kwargs);
  } else {
    o = py::none();
  }
  return o;
}

template<typename F, typename... Args>
decltype(auto) applyArgs(
    std::optional<GilWrapper<py::args>>& args, std::optional<GilWrapper<py::kwargs>>& kwargs, F&& f, py::object& o,
    Args&&... moreargs) {
  if constexpr (std::is_same_v<std::decay_t<F>, py::function>) {
    if (args) {
      if (kwargs) {
        py::tuple tup = py::reinterpret_steal<py::tuple>(o.release());
        return f(std::forward<Args>(moreargs)..., *tup[0], **tup[1]);
      } else {
        return f(std::forward<Args>(moreargs)..., *o);
      }
    } else if (kwargs) {
      return f(std::forward<Args>(moreargs)..., **o);
    } else {
      return f(std::forward<Args>(moreargs)...);
    }
  } else {
    if (args) {
      if (kwargs) {
        py::tuple tup = py::reinterpret_steal<py::tuple>(o.release());
        return f(
            std::forward<Args>(moreargs)..., py::reinterpret_borrow<py::args>(tup[0]),
            py::reinterpret_borrow<py::kwargs>(tup[1]));
      } else {
        return f(std::forward<Args>(moreargs)..., py::reinterpret_borrow<py::args>(o));
      }
    } else if (kwargs) {
      return f(std::forward<Args>(moreargs)..., py::reinterpret_borrow<py::kwargs>(o));
    } else {
      return f(std::forward<Args>(moreargs)...);
    }
  }
}

struct ApplyToQueue {
  QueueWrapper& q;
  ApplyToQueue(QueueWrapper& q) : q(q) {}
  template<typename Callback>
  void operator()(Callback&& callback) {
    q.enqueue(std::forward<Callback>(callback), {}, {});
  }
  template<typename Callback>
  void operator()(Callback&& callback, py::args args) {
    q.enqueue(std::forward<Callback>(callback), std::move(args), {});
  }
  template<typename Callback>
  void operator()(Callback&& callback, py::kwargs kwargs) {
    q.enqueue(std::forward<Callback>(callback), {}, std::move(kwargs));
  }
  template<typename Callback>
  void operator()(Callback&& callback, py::args args, py::kwargs kwargs) {
    q.enqueue(std::forward<Callback>(callback), std::move(args), std::move(kwargs));
  }
};

} // namespace

struct RpcWrapper {
  std::shared_ptr<RpcCounted> rpc;

  RpcWrapper() {
    moduleIncRef();
    rpc = std::shared_ptr<RpcCounted>(new RpcCounted(), [](RpcCounted* ptr) noexcept {
      if (PyGILState_Check()) {
        py::gil_scoped_release gil;
        delete ptr;
      } else {
        delete ptr;
      }
    });
    rpc->init(rpc);
    std::call_once(importThreadingFlag, importThreading);
  }
  ~RpcWrapper() {
    rpc.reset();
    moduleDecRef();
  }

  void setName(std::string_view name) {
    rpc->setName(name);
  }
  std::string_view getName() const {
    return rpc->getName();
  }
  void listen(std::string_view addr) {
    rpc->listen(addr);
  }
  void connect(std::string_view addr) {
    rpc->connect(addr);
  }

  void define(std::string_view name, py::function callback, py::kwargs kwargs) {
    if (kwargs.contains("batch_size") && kwargs["batch_size"].ptr() != Py_None) {
      using MyBatcher = Batcher<rpc::RpcDeferredReturn<GilWrapper<py::object>>>;
      MyBatcher batcher;
      int batchSize = py::cast<int>(kwargs["batch_size"]);
      if (kwargs.contains("device")) {
        std::string device = py::cast<std::string>(kwargs["device"]);
        batcher = MyBatcher(batchSize, device, 0);
      } else {
        batcher = MyBatcher(batchSize);
      }
      rpc->define<GilWrapper<py::object>(std::optional<GilWrapper<py::args>>, std::optional<GilWrapper<py::kwargs>>)>(
          name, [batcher = std::move(batcher), callback = GilWrapper<py::function>(std::move(callback))](
                    rpc::RpcDeferredReturn<GilWrapper<py::object>> returnCallback,
                    std::optional<GilWrapper<py::args>> args, std::optional<GilWrapper<py::kwargs>> kwargs) mutable {
            py::gil_scoped_acquire gil;
            if (_Py_IsFinalizing()) {
              return;
            }
            keepThreadAlive();
            auto retval = batcher.stack(getBatchValue(args, kwargs), std::move(returnCallback));
            if (retval) {
              batcher.unbatch(applyArgs(args, kwargs, *callback, retval->first), retval->second);
            }
          });
    } else {
      rpc->define<GilWrapper<py::object>(std::optional<GilWrapper<py::args>>, std::optional<GilWrapper<py::kwargs>>)>(
          name, [callback = GilWrapper<py::function>(std::move(callback))](
                    std::optional<GilWrapper<py::args>> args, std::optional<GilWrapper<py::kwargs>> kwargs) mutable {
            py::gil_scoped_acquire gil;
            if (_Py_IsFinalizing()) {
              return GilWrapper<py::object>(py::none());
            }
            keepThreadAlive();
            if (args) {
              if (kwargs) {
                return GilWrapper<py::object>((*callback)(**std::move(*args), ***std::move(*kwargs)));
              } else {
                return GilWrapper<py::object>((*callback)(**std::move(*args)));
              }
            } else if (kwargs) {
              return GilWrapper<py::object>((*callback)(***std::move(*kwargs)));
            } else {
              return GilWrapper<py::object>((*callback)());
            }
          });
    }
  }

  void defineDeferred(std::string_view name, py::function callback, py::kwargs kwargs) {
    if (kwargs.contains("batch_size") && kwargs["batch_size"].ptr() != Py_None) {
      using MyBatcher = Batcher<rpc::RpcDeferredReturn<GilWrapper<py::object>>>;
      std::shared_ptr<MyBatcher> batcher;
      int batchSize = py::cast<int>(kwargs["batch_size"]);
      if (kwargs.contains("device")) {
        std::string device = py::cast<std::string>(kwargs["device"]);
        batcher = std::make_shared<MyBatcher>(batchSize, device, 0);
      } else {
        batcher = std::make_shared<MyBatcher>(batchSize);
      }
      rpc->define<GilWrapper<py::object>(std::optional<GilWrapper<py::args>>, std::optional<GilWrapper<py::kwargs>>)>(
          name, [batcher = std::move(batcher), callback = GilWrapper<py::function>(std::move(callback))](
                    rpc::RpcDeferredReturn<GilWrapper<py::object>> returnCallback,
                    std::optional<GilWrapper<py::args>> args, std::optional<GilWrapper<py::kwargs>> kwargs) mutable {
            py::gil_scoped_acquire gil;
            if (_Py_IsFinalizing()) {
              return;
            }
            keepThreadAlive();
            auto retval = batcher->stack(getBatchValue(args, kwargs), std::move(returnCallback));
            if (retval) {
              py::object o = std::move(retval->first);
              auto returnCallback2 = py::cpp_function([batcher, retval = std::move(retval)](py::object r) mutable {
                batcher->unbatch(std::move(r), retval->second);
              });
              applyArgs(args, kwargs, *callback, o, std::move(returnCallback2));
            }
          });
    } else {
      rpc->define<GilWrapper<py::object>(std::optional<GilWrapper<py::args>>, std::optional<GilWrapper<py::kwargs>>)>(
          name, [callback = GilWrapper<py::function>(std::move(callback))](
                    rpc::RpcDeferredReturn<GilWrapper<py::object>> returnCallback,
                    std::optional<GilWrapper<py::args>> args, std::optional<GilWrapper<py::kwargs>> kwargs) mutable {
            py::gil_scoped_acquire gil;
            if (_Py_IsFinalizing()) {
              return;
            }
            keepThreadAlive();
            if (args) {
              if (kwargs) {
                (*callback)(std::move(returnCallback), **std::move(*args), ***std::move(*kwargs));
              } else {
                (*callback)(std::move(returnCallback), **std::move(*args));
              }
            } else if (kwargs) {
              (*callback)(std::move(returnCallback), ***std::move(*kwargs));
            } else {
              (*callback)(std::move(returnCallback));
            }
          });
    }
  }

  std::shared_ptr<QueueWrapper> defineQueue(std::string_view name, py::kwargs kwargs) {
    auto q = std::make_shared<QueueWrapper>();
    q->timeout = rpc->getTimeout();

    if (kwargs.contains("batch_size") && kwargs["batch_size"].ptr() != Py_None) {
      const int batchSize = py::cast<int>(kwargs["batch_size"]);

      if (kwargs.contains("dynamic_batching") && kwargs["dynamic_batching"].ptr() != Py_None &&
          py::cast<bool>(kwargs["dynamic_batching"])) {
        q->batchSize = batchSize;
        rpc->define<GilWrapper<py::object>(std::optional<GilWrapper<py::args>>, std::optional<GilWrapper<py::kwargs>>)>(
            name,
            [q](rpc::RpcDeferredReturn<GilWrapper<py::object>> returnCallback, std::optional<GilWrapper<py::args>> args,
                std::optional<GilWrapper<py::kwargs>> kwargs) mutable noexcept {
              q->enqueue(std::move(returnCallback), std::move(args), std::move(kwargs));
            });
      } else {
        using MyBatcher = Batcher<rpc::RpcDeferredReturn<GilWrapper<py::object>>>;
        std::shared_ptr<MyBatcher> batcher;
        if (kwargs.contains("device")) {
          std::string device = py::cast<std::string>(kwargs["device"]);
          batcher = std::make_shared<MyBatcher>(batchSize, device, 0);
        } else {
          batcher = std::make_shared<MyBatcher>(batchSize);
        }
        rpc->define<GilWrapper<py::object>(std::optional<GilWrapper<py::args>>, std::optional<GilWrapper<py::kwargs>>)>(
            name, [batcher = std::move(batcher),
                   q](rpc::RpcDeferredReturn<GilWrapper<py::object>> returnCallback,
                      std::optional<GilWrapper<py::args>> args, std::optional<GilWrapper<py::kwargs>> kwargs) mutable {
              py::gil_scoped_acquire gil;
              if (_Py_IsFinalizing()) {
                return;
              }
              keepThreadAlive();
              auto retval = batcher->stack(getBatchValue(args, kwargs), std::move(returnCallback));
              if (retval) {
                py::object o = std::move(retval->first);
                rpc::RpcDeferredReturn<GilWrapper<py::object>> returnCallback2;
                returnCallback2.f = [batcher, retval = std::move(retval)](GilWrapper<py::object> r) mutable {
                  batcher->unbatch(std::move(*r), retval->second);
                };
                applyArgs(args, kwargs, ApplyToQueue(*q), o, std::move(returnCallback2));
              }
            });
      }
    } else {
      rpc->define<GilWrapper<py::object>(std::optional<GilWrapper<py::args>>, std::optional<GilWrapper<py::kwargs>>)>(
          name,
          [q](rpc::RpcDeferredReturn<GilWrapper<py::object>> returnCallback, std::optional<GilWrapper<py::args>> args,
              std::optional<GilWrapper<py::kwargs>> kwargs) mutable noexcept {
            q->enqueue(std::move(returnCallback), std::move(args), std::move(kwargs));
          });
    }
    return q;
  }

  auto async(
      std::string_view peerName, std::string_view functionName, std::optional<py::args> args,
      std::optional<py::kwargs> kwargs) {
    auto buffer = rpc->serializeArguments(args, kwargs);
    py::gil_scoped_release gil;
    Promise<FutureWrapper> promise;
    auto future = promise.getFuture();
    rpc->asyncCallback<GilWrapper<py::object>>(
        peerName, functionName,
        [promise = std::move(promise)](GilWrapper<py::object>* r, rpc::Error* error) mutable noexcept {
          if (r) {
            promise.setResult(std::move(*r));
          } else {
            promise.setException(std::move(*error));
          }
        },
        buffer);
    return future;
  }

  auto async_noargs(std::string_view peerName, std::string_view functionName) {
    return async(peerName, functionName, {}, {});
  }
  auto async_args(std::string_view peerName, std::string_view functionName, py::args args) {
    return async(peerName, functionName, args, {});
  }
  auto async_kwargs(std::string_view peerName, std::string_view functionName, py::kwargs kwargs) {
    return async(peerName, functionName, {}, kwargs);
  }
  auto async_args_kwargs(std::string_view peerName, std::string_view functionName, py::args args, py::kwargs kwargs) {
    return async(peerName, functionName, args, kwargs);
  }

  void asyncCallback(
      std::string_view peerName, std::string_view functionName, py::function callback, std::optional<py::args> args,
      std::optional<py::kwargs> kwargs) {
    auto buffer = rpc->serializeArguments(args, kwargs);
    py::gil_scoped_release gil;
    rpc->asyncCallback<GilWrapper<py::object>>(
        peerName, functionName,
        [callback = GilWrapper<py::function>(std::move(callback))](
            GilWrapper<py::object>* r, rpc::Error* error) mutable noexcept {
          py::gil_scoped_acquire gil;
          if (_Py_IsFinalizing()) {
            return;
          }
          keepThreadAlive();
          if (r) {
            std::move (*callback)(std::move(**r), py::none());
          } else {
            std::move (*callback)(py::none(), py::str(error->what()));
          }
        },
        buffer);
  }

  auto asyncCallback_noargs(std::string_view peerName, std::string_view functionName, py::function callback) {
    return asyncCallback(peerName, functionName, std::move(callback), {}, {});
  }
  auto
  asyncCallback_args(std::string_view peerName, std::string_view functionName, py::function callback, py::args args) {
    return asyncCallback(peerName, functionName, std::move(callback), args, {});
  }
  auto asyncCallback_kwargs(
      std::string_view peerName, std::string_view functionName, py::function callback, py::kwargs kwargs) {
    return asyncCallback(peerName, functionName, std::move(callback), {}, kwargs);
  }
  auto asyncCallback_args_kwargs(
      std::string_view peerName, std::string_view functionName, py::function callback, py::args args,
      py::kwargs kwargs) {
    return asyncCallback(peerName, functionName, std::move(callback), args, kwargs);
  }

  py::object sync(
      std::string_view peerName, std::string_view functionName, std::optional<py::args> args,
      std::optional<py::kwargs> kwargs) {
    auto buffer = rpc->serializeArguments(args, kwargs);
    py::gil_scoped_release gil;
    return *rpc->sync<GilWrapper<py::object>>(peerName, functionName, buffer);
  }

  py::object sync_noargs(std::string_view peerName, std::string_view functionName) {
    return sync(peerName, functionName, {}, {});
  }
  py::object sync_args(std::string_view peerName, std::string_view functionName, py::args args) {
    return sync(peerName, functionName, args, {});
  }
  py::object sync_kwargs(std::string_view peerName, std::string_view functionName, py::kwargs kwargs) {
    return sync(peerName, functionName, {}, kwargs);
  }
  py::object
  sync_args_kwargs(std::string_view peerName, std::string_view functionName, py::args args, py::kwargs kwargs) {
    return sync(peerName, functionName, args, kwargs);
  }

  void setTimeout(float seconds) {
    rpc->setTimeout(std::chrono::milliseconds(int(seconds * 1000)));
  }

  void debugInfo() {
    rpc->debugInfo();
  }

  void setTransports(std::vector<std::string> names) {
    rpc->setTransports(names);
  }

  operator std::shared_ptr<rpc::Rpc>() const {
    return rpc;
  }
};

struct AllReduceWrapper : FutureWrapper {
  AllReduce h;
  AllReduceWrapper() = default;
  AllReduceWrapper(AllReduce h) : h(std::move(h)) {}

  void cancel() {
    h = {};
    FutureWrapper::cancel();
  }
};

struct GroupWrapper : Group {
  GroupWrapper(std::shared_ptr<rpc::Rpc> rpc, std::string groupName) : Group(std::move(rpc), std::move(groupName)) {}
  std::shared_ptr<AllReduceWrapper> allReduce(std::string name, py::object data, py::object op = py::none()) {
    py::gil_scoped_release gil;
    Promise<AllReduceWrapper> promise;
    auto future = promise.getFuture();

    if (!op.is_none()) {
      auto h = Group::allReduce(
          std::move(name), GilWrapper<py::object>(std::move(data)),
          [op = GilWrapper<py::object>(op)](GilWrapper<py::object>& a, GilWrapper<py::object>& b) mutable {
            py::gil_scoped_acquire gil;
            if (_Py_IsFinalizing()) {
              return;
            }
            a = (*op)(*a, *b);
          },
          [promise = std::move(promise)](GilWrapper<py::object>* r, rpc::Error* error) mutable noexcept {
            if (r) {
              py::gil_scoped_acquire gil;
              if (_Py_IsFinalizing()) {
                return;
              }
              promise.setResult(std::move(*r));
            } else {
              promise.setException(std::move(*error));
            }
          });
      future->h = std::move(h);
      return future;
    }

    if (auto t = rpc::tryFromPython(data)) {
      auto h = Group::allReduce(
          std::move(name), *t, ReduceSum(),
          [promise = std::move(promise)](rpc::Tensor* r, rpc::Error* error) mutable noexcept {
            if (r) {
              py::gil_scoped_acquire gil;
              if (_Py_IsFinalizing()) {
                return;
              }
              promise.setResult(rpc::toPython(*r));
            } else {
              promise.setException(std::move(*error));
            }
          });
      future->h = std::move(h);
      return future;
    }
    if (py::isinstance<py::list>(data)) {
    }
    fatal("all_reduce can only use a builtin operator with Tensor or list of Tensors. Please specify an operator "
          "function");
  }
};

struct EnvPoolWrapper {
  EnvPool envPool;
  int batchSize_;
  int numBatches_;
  std::unique_ptr<EnvStepper> stepper;
  EnvPoolWrapper(py::object envInit, int numProcesses, int batchSize, int numBatches) : envPool(envInit, numProcesses) {
    batchSize_ = batchSize;
    numBatches_ = numBatches;
    stepper = envPool.spawn();
  }
  EnvStepperFuture step(int batchIndex, py::object action) {
    if (auto tensor = rpc::tryFromPython(action); tensor && tensor->dim() == 1) {
      if (tensor->size(0) != batchSize_) {
        throw std::runtime_error(fmt::sprintf(
            "env step was passed an action tensor with batch size %d, expected %d", tensor->size(0), batchSize_));
      }
    }
    if (batchIndex < 0 || batchIndex >= numBatches_) {
      throw std::runtime_error(fmt::sprintf(
          "env step was passed an out-of-range batch index %d (valid range is [0,%d))", batchIndex, 0, numBatches_));
    }
    return stepper->step(batchIndex, action);
  }
};

struct BatcherWrapper {
  Batcher<char> batcher;

  rpc::SpinMutex mutex;
  std::deque<py::object> queue;
  rpc::Semaphore sem;
  std::deque<Promise<FutureWrapper>> waiters;

  BatcherWrapper(int64_t batchSize, std::string device, int64_t batchDimension)
      : batcher(batchSize, device, batchDimension) {}

  void enqueue(py::object value) {
    {
      std::unique_lock l(mutex);
      if (!waiters.empty()) {
        auto promise = std::move(waiters.front());
        waiters.pop_front();
        l.unlock();
        promise.setResult(std::move(value));
        return;
      }
      queue.push_back(std::move(value));
    }
    sem.post();
  }

  py::object await() {
    Promise<FutureWrapper> promise;
    py::object r = promise.getFuture()->await();
    {
      std::unique_lock l(mutex);
      if (!queue.empty()) {
        auto r = std::move(queue.front());
        queue.pop_front();
        l.unlock();
        promise.setResult(std::move(r));
      } else {
        waiters.push_back(std::move(promise));
      }
    }
    return r;
  }

  bool empty() {
    std::unique_lock l(mutex);
    return queue.empty();
  }

  py::object get() {
    std::unique_lock l(mutex);
    if (!queue.empty()) {
      auto r = std::move(queue.front());
      queue.pop_front();
      return r;
    } else {
      Promise<FutureWrapper> promise;
      auto future = promise.getFuture();
      waiters.push_back(std::move(promise));
      l.unlock();
      return future->result();
    }
  }

  void stack(py::object data) {
    auto retval = batcher.stack(std::move(data), {});
    if (retval) {
      enqueue(std::move(retval->first));
    }
  }
  void cat(py::object data) {
    batcher.cat(std::move(data), [this](py::object value) { enqueue(value); });
  }
};

void set_log_level(py::object level) {
  if (py::isinstance<py::str>(level)) {
    std::string str = level.cast<std::string>();
    if (str == "none") {
      currentLogLevel = LOG_NONE;
    } else if (str == "error") {
      currentLogLevel = LOG_ERROR;
    } else if (str == "info") {
      currentLogLevel = LOG_INFO;
    } else if (str == "verbose") {
      currentLogLevel = LOG_VERBOSE;
    } else if (str == "debug") {
      currentLogLevel = LOG_DEBUG;
    } else {
      fatal("Unknown log level '%s'", str);
    }
  } else {
    currentLogLevel = (LogLevel)level.cast<int>();
  }
}

PYBIND11_MODULE(_C, m) {
  moduleIncRef();
  rpc::moolibModule = m.ptr();
  m.add_object("_cleanup", py::capsule([] {
                 moduleUnload();
                 rpc::moolibModule = nullptr;
               }));

  // pybind signatures don't play well with Sphinx
  py::options options;
  options.disable_function_signatures();

  m.doc() = py::doc(R"d(
    A communications library for distributed ML training.

    moolib offers general purpose RPC with automatic transport selection
    (shared memory, tcp/ip, infiniband) allowing models to data-parallelise
    their training and synchronize gradients and model weights across many nodes.
  )d");

  // py::register_exception<rpc::Error>(m, "RpcError");
  m.def(
      "set_logging", [](py::object logging) { pyLogging = logging; }, py::doc(R"d(
    Set logging using the python logging module.

    Args:
        logging (logging):
  )d"));
  m.def("set_log_level", set_log_level, py::doc(R"d(
    Set the level to log at.

    Args:
        level (int): The level to log at.

  )d"));
  m.def("create_uid", &randomName, py::doc(R"d(
    Generate a unique user id.

    Returns:
        uuid (str): a unique id that can be used for a broker group.

  )d"));
  m.def(
      "set_max_threads", [](int n) { rpc::scheduler.setMaxThreads(n); }, py::doc(R"d(
    Set the maximum number of threads used by the moolib.

    Args:
        max_threads (int): the maximum number of threads.
  )d"));
  py::class_<EnvRunner>(m, "EnvRunner", "Docstring for EnvRunner")
      .def("start", &EnvRunner::start, py::doc(R"d(
        Documentation for start.
      )d"))
      .def("running", &EnvRunner::running, py::doc(R"d(
        Documentation for running.
      )d"));
  py::class_<EnvPoolWrapper>(m, "EnvPool", R"d(
    A class to run sets of gym-like environments in different processes.

    This class will often be used in an RL setting, see ``examples/impala/impala.py``, with batching
    happening in a background thread.

    The class maintains ``num_batches`` batches of environments, with ``batch_size`` environments each.
    This means the batches can be stepped through alternately, for increased efficiency
    (cf "double-buffering"), and the whole ``EnvPool`` uses ``num_process`` to run these environments.

    Example:
        Provide and example here::

            def create_env():
                return gym.make('NetHackChallenge-v0')

            batch_size = 32
            n_actions = create_env().action_space.n

            # step through two sets of envs, each with its own process ("double-buffering")
            batcher = EnvPool(create_env, num_processes=64, batch_size=batch_size, num_batches=2)

            for i in range(20):
                actions = (torch.rand(batch_size) * n_actions).int()
                batcher.step(i % 2, actions)
  )d")
      .def(
          py::init<py::object, int, int, int>(), py::arg("create_env"), py::arg("num_processes"), py::arg("batch_size"),
          py::arg("num_batches"), py::doc(R"d(
            Init.

            Args:
                create_env (Callable[[], gym.Env]): a user-defined function that returns a Gym environment.
                num_processes (int): how many processes should be used for running environments.
                batch_size (int): the number of environments in one batch.
                num_batches (int): the number of batches to maintain (for double-buffering).
      )d"))
      .def("step", &EnvPoolWrapper::step, py::arg("batch_index"), py::arg("action"), py::doc(R"d(
        Step through a batch of envs.

        Args:
            batch_index (int): index of the batch of envs are we stepping.
            action (torch.Tensor): actions for each of the envs [BATCH_SIZE].
      )d"));
  py::class_<EnvStepper>(m, "EnvStepper", R"d(
    A helper class for EnvPool.

    This class is used by :class:`EnvPool` implementation to step through its batches of environments via RPC.
  )d")
      .def("step", &EnvStepper::step, py::arg("batch_index"), py::arg("action"), py::doc(R"d(
    Step through a batch of envs.

    Args:
        batch_index (int): index of the batch of envs are we stepping.
        action (torch.Tensor): actions for each of the envs [BATCH_SIZE]
  )d"));
  py::class_<EnvStepperFuture>(m, "EnvStepperFuture", "A future result from an EnvStepper step.")
      .def("result", &EnvStepperFuture::result, py::doc(R"d(Get the result of the Future.)d"));
  py::class_<Accumulator>(m, "Accumulator", R"d(
        Accumulate and synchronize gradients and state from multiple peers in the cohort.

        This class allows for the accumulation and synchronization of gradients and model weights
        during distributed training.  The communication requires an underlying :class:`Group` object,
        which can either be  passed into the constructor, or will be created by the accumulator
        during initialisation.

        Many of the calls on the Accumulator involve synchronisation across the network. This occurs
        in the ``update()`` call, that must be called regularly, throughout training.

        Example:
            The accumulator takes a moolib :class:`Group` to coordinate::

                model = ...
                peer = moolib.Rpc()
                group = moolib.Group(peer, "foo_group"))
                accumulator = moolib.Accumulator("bar", model.parameters(), model.buffers(), group=group)
                accumulator.connect(ADDRESS)

                opt = torch.optim.Adam(model.parameters(), lr=0.001)

                while True:
                    # Update the status of the objects across the network
                    group.update()
                    accumulator.update()

                    # 0. Check we are connected.
                    if not accumulator.connected():
                        time.sleep(0.25)
                        continue

                    # 1. Synchronize our state.
                    if accumulator.wants_state():
                        accumulator.set_state({"optimizer": opt.state_dict()})
                    if accumulator.has_new_state():
                        opt.load_state_dict(accumulator.state()["optimizer"])

                    # 2. Generate our gradients then reduce them.
                    if accumulator.wants_gradients():
                        y_pred = model(X)
                        loss = ...
                        loss.backward()
                        accumulator.reduce_gradients(batch_size) # has_gradients() -> True after this

                    # 3. Step our gradients then reset.
                    if accumulator.has_gradients():
                        opt.step()
                        accumulator.zero_gradients()  # has_gradients() -> False after this
      )d")
      .def(
          py::init<std::string, py::object, py::object, const GroupWrapper*>(), py::arg("name"), py::arg("parameters"),
          py::arg("buffers"), py::arg("group") = py::none(), py::doc(R"d(
        Init.

        The ``Accumulator`` performs its operations on an underlying :class:`Group` object. This can either
        be passed in in the `group` keyword, or will be constructed by the `Accumulator`, with a group
        name based on the `Accumulator`'s name.

        Args:
            name (str): the name of the accumulator.
            parameters (Iterable[torch.Parameter]): the parameters to accumulate - eg ``model.parameters()``.
            buffers (Iterable[torch.Tensor]): the buffers to accumulate - eg ``model.buffers()``.
            group (Optional[moolib.Group]): a :class:`Group` on which to perform accumulation. If None,
                a new :class:`Group` will be constructed, with a name based on ``name`` parameter.
          )d"))
      .def("connect", &Accumulator::connect, py::arg("address"), py::doc(R"d(
        Connect the underlying Rpc object to a :class:`Broker` at a given network address.

        This is a non-blocking call, and moolib will maintain this connection by regularly reconnecting
        if the connection is lost.

        Args:
            address (str): address to connect to (eg: ``"127.0.0.1:1234"``)
      )d"))
      .def("update", &Accumulator::update, py::doc(R"d(
        Update the state of the accumulator across the network.

        This call executes the communication of the accumulator with the network, including connecting to the
        network, sending or receiving state updates.
      )d"))
      .def("connected", &Accumulator::connected, py::doc(R"d(
        Check whether the underlying :class:`Group` is connected to a :class:`Broker` and is ready to train.

        Returns:
            bool
      )d"))
      .def("wants_state", &Accumulator::wantsState, py::doc(R"d(
        Returns ``True`` if the accumulator has requested a new state to sync.

        This is required for new peers to get their initial state to start training, and is used to
        ensure synchronization of the state during training. The new state can be set using ``set_state``.

        Returns:
            bool
      )d"))
      .def("has_new_state", &Accumulator::hasNewState, py::doc(R"d(
        Returns ``True`` if the accumulator been given a new state to sync.

        Returns:
            bool
      )d"))
      .def("has_gradients", &Accumulator::hasGradients, py::doc(R"d(
        Returns ``True`` if the accumulator has gradients that have been reduced.

        This will return True after a call to ``reduce_gradients`` or ``skip_gradients`` has been executed.

        Returns:
            bool
      )d"))
      .def("wants_gradients", &Accumulator::wantsGradients, py::doc(R"d(
        Returns ``True`` if the accumulator is ready to reduce more gradients or skip gradients.

        Returns:
            bool
      )d"))
      .def("set_state", &Accumulator::setState, py::arg("state"), py::doc(R"d(
        Set the user state for synchronization.

        This state is the user-defined state that must be synchronized for training to be successful.
        This often includes the model weights and optimizer ``state_dicts``, along with any other objects.

        This function accepts any objects that can be ``pickled``. These objects will what is returned in
        a call to ``state``.

        Args:
            state (object): The current user-defined state to synchronize.
      )d"))
      .def("state", &Accumulator::state, py::doc(R"d(
        Returns the current user state.

        See ``set_state`` for more details on what is returned.

        Returns:
            Object: The state (as set in ``set_state``)
      )d"))
      .def("skip_gradients", &Accumulator::skipGradients, py::doc(R"d(
        Force the accumulator to skip this round of gradient reductions.

        At each step, if ``wants_gradients`` returns true, either ``reduce_gradients`` or ``skip_gradients``
        must be called.
      )d"))
      .def("reduce_gradients", &Accumulator::reduceGradients, py::arg("batch_size"), py::doc(R"d(
        Reduce sum of the gradients across all the accumulators.

        At each step, if ``wants_gradients`` returns true, either ``reduce_gradients`` or ``skip_gradients``
        must be called.

        Args:
            batch_size (int): The size of the virtual batch_size to reduce to.
      )d"))
      .def("zero_gradients", &Accumulator::zeroGradients, py::doc(R"d(
        Reset the gradients of the accumulator to zero.
      )d"))
      .def("model_version", &Accumulator::modelVersion, py::doc(R"d(
        Return the current model version.
        This number is automatically incremented when ``zero_gradients`` is called.
        The peer with the highest model version will be selected as the leader.

        Returns:
            n (int): the number of updates.
      )d"))
      .def("set_model_version", &Accumulator::setModelVersion, py::arg("n"), py::doc(R"d(
         Set the current model version.

         The default value is:

         Args:
             n (int): the virtual batch size.
       )d"))
      .def("set_virtual_batch_size", &Accumulator::setVirtualBatchSize, py::arg("n"), py::doc(R"d(
        Set the virtual batch size of the reduction operation.

        The default value is:

        Args:
            n (int): the virtual batch size.
      )d"))
      .def("set_parallel_gradients", &Accumulator::setParallelGradients, py::arg("n"), py::doc(R"d(
        Set the number of parallel gradient reduction operations.

        The default value is:

        Args:
            n (int): the number of parallel gradient operations.
      )d"))
      .def("get_leader", &Accumulator::getLeader, py::doc(R"d(
        Returns the name of the leader in the group.

        Returns:
            (str): returns the name of the leader in the group.
      )d"))
      .def("is_leader", &Accumulator::isLeader, py::doc(R"d(
               Return true iff this peer is the leader of the group.

               Returns:
                   (bool): whether this peer is the leader of the group.
             )d"))
      .def("get_gradient_stats", &Accumulator::getGradientStats, py::doc(R"d(
        Return a dict of statistics about the gradients syncing process.

        This include information about ``batch_size``, ``num_updates`` and ``num_skipped``.

        Returns:
            (dict): statistics about the gradient syncing.
      )d"));
  py::class_<BatcherWrapper>(m, "Batcher", R"d(
    A auxiliary class to asynchronously batch tensors into an chosen batch size on a certain device.

    This class will often be used in an RL setting, see ``examples/impala/impala.py``, with batching
    happening in a background thread. This class is ``awaitable`` with asyncio.

    Example:
        The batcher can take a series of data points and return a batched version::

            batcher = Batcher(size=32, device="cuda:0", dim=0)

            datapoint = {
              "batchme": torch.ones([2, 7]),
              "pliss": torch.randn([3])
            }

            while True:
                batcher.cat(data_point)
                while not batcher.empty():
                    mb = learn_batcher.get()

                    mb["batchme"]  # [32, 2, 7]
                    mb["pliss"]  # [32, 3]
  )d")
      .def(
          py::init<int64_t, std::string, int64_t>(), py::arg("size"), py::arg("device") = "cpu", py::arg("dim") = 0,
          py::doc(R"d(
        Init.

        Args:
            size (int): the batch size to batch to.
            device (str): the device to batch tensors on.
            dim (int): the tensor dim to perform the operations along.
      )d"))
      .def("stack", &BatcherWrapper::stack, py::arg("tensors"), py::doc(R"d(
        Batch the tensors by stacking along the target dim.

        Args:
            tensors: the tensors to stack.
      )d"))
      .def("cat", &BatcherWrapper::cat, py::arg("tensors"), py::doc(R"d(
        Batch the tensors by concatenating along the target dim.

        Args:
            tensors: the tensors to concatenate.
      )d"))
      .def("empty", &BatcherWrapper::empty, py::doc(R"d(
        Returns True if there are no batched tensors to ``get``.

        Returns:
            bool:
      )d"))
      .def("get", &BatcherWrapper::get, py::doc(R"d(
        Return a batched tensor.

        This is a blocking call.

        Returns:
            batched tensor
      )d"))
      .def("__await__", &BatcherWrapper::await);
  py::class_<rpc::RpcDeferredReturn<GilWrapper<py::object>>>(m, "RpcDeferredReturn", R"d(
    A deferred return from call to a ``define_deferred`` function.
  )d")
      .def("__call__", &rpc::RpcDeferredReturn<GilWrapper<py::object>>::operator()<const py::object&>);
  py::class_<QueueWrapper, std::shared_ptr<QueueWrapper>>(m, "Queue", R"d(
    A tensorpipe Queue class.
  )d")
      .def(py::init<>())
      .def("enqueue", &QueueWrapper::enqueue, py::doc(R"d(
        Add an object to the Queue.

        Args:
            object (Object): add this to the queue.
      )d"))
      .def("__await__", &QueueWrapper::await)
      .def("__iter__", &QueueWrapper::await);
  py::class_<RpcWrapper>(m, "Rpc", R"d(
    A class to execute Remote Procedure Calls.

    The class represents one peer in a cohort, and allows you to define functions to be called by
    other peers, or to call functions that other peers have defined.

    Peers are referred to by name (as opposed to a network address) though an address is required
    for an initial connection, and there is an built-in discovery service to allow peers to find
    each other, so long as there is connection path between them.

    Example:
        Using the ``Rpc`` class is quite simple::

            import moolib

            def foo(str):
                print(str)
                return 42

            host = moolib.Rpc()
            host.set_name("host")
            host.define("bar", foo)
            host.listen("127.0.0.1:1234")

            client = moolib.Rpc()
            client.connect("127.0.0.1:1234")

            future = client.async_("host", "bar", "hello world")
            print(future.get())

        The output would then be::

            hello world
            42
    )d")
      .def(py::init<>())
      .def("set_name", &RpcWrapper::setName, py::arg("name"), py::doc(R"d(
        Set the unique name of the peer.

        The name is a unique identifier of the peer within the current cohort of connected peers.
        The name chosen here must not exist among other peers in the cohort, and this function must be
        called before attempting to join the network (ie  any ``listen`` or ``connect`` calls).

        Args:
            name (str): the name of the peer.
      )d"))
      .def("get_name", &RpcWrapper::getName, py::doc(R"d(
        Return the name of the peer.

        The name is a unique identifier of the peer within the current cohort of connected peers.
        In the default case this is randomly generated string which is guaranteed to be globally unique,
        or it can be actively set by calling ``set_name``.

        Returns:
            str: the name of the peer (as set by ``set_name`` or default).
      )d"))
      .def("listen", &RpcWrapper::listen, py::arg("address"), py::doc(R"d(
        Listen on a network address.

        This is a non-blocking call.

        Args:
            address (str): address to listen to (eg: ``"127.0.0.1:1234"``).
      )d"))
      .def("connect", &RpcWrapper::connect, py::arg("address"), py::doc(R"d(
        Connect to a network address.

        This is a non-blocking call, and moolib will maintain this connection by regularly reconnecting
        if the connection is lost.

        Args:
            address (str): address to connect to (eg: ``"127.0.0.1:1234"``).
      )d"))
      .def("define", &RpcWrapper::define, py::doc(R"d(
        Define a function to be available for peers to call.

        This function will be run asynchronously, called from an internal thread. The maximum number
        of concurrent functions that can be run is limited by the maximum number of threads (see
        ``moolib.set_max_threads``)

        This function accepts two keyword arguments:
            * ``batch_size`` (``int``)
                any tensors in any arguments of calls to this function will be
                batched to this batch_size before remote execution, and split back on return.
            * ``device`` (``str``)
                any tensors in any arguments of calls to this function will be batched
                on this device before remote execution, and returns to the original device.

        Note:
            It is best practice to call ``define`` methods before connecting to the network, to avoid race conditions.

        Args:
            name (str): a unique name for the function, for peers to use.
            function (Callable[[], None]): a Python function, for peers to call.
            kwargs: specifications for how to treat Tensor arguments to the function.
      )d"))
      .def("define_deferred", &RpcWrapper::defineDeferred, py::doc(R"d(
        Define a deferred function to be available for peers to call.

        The deferred function must accept a callback provided by moolib as its first argument.
        This defined function will not return until this callback is called, and the function
        returns the argument to the callback.

        see ``define()`` for keyword arguments and Notes

        Args:
            name (str): a unique name for the function, for peers to use.
            callback (Callable[[], None]): a Python function, for peers to call.
            kwargs: see ``define()`` for keyword arguments.
      )d"))
      .def("define_queue", &RpcWrapper::defineQueue, py::doc(R"d(
        Define a queue to be available for peers to populate, when they call the function_name.

        This queue will be populated on each "function call" by peers, and will be populated with
        a moolib provided callback and the arguments to the function call. The "function call"
        will be not return until the callback is called, and returns the arguments to the callback.

        see ``define()`` for keyword arguments and Notes

        Example:

            Usage::

                # host.py
                # ... setup host ...
                queue = host.define_queue("foo")

                while True:
                    cb, arg1, arg2 = queue.result()
                    print(arg1, arg2)
                    cb(42) # returns 42 to client

                # client.py
                # ... setup client ...
                print(host.sync("foo", arg1, arg2))  # eventually prints 42

        Args:
            name (str): a unique name for the function, for peers to use.
            kwargs: see ``define()`` for keyword arguments.
      )d"))
      .def("async_", &RpcWrapper::async_args_kwargs, py::doc(R"d(
        Make an asynchronous call to ``function_name`` and return a ``Future``.

        Note:
            If you send a Tensor as an argument, you **may not** modify the Tensor data until the
            result is returned.

        Args:
            name (str): the name of the peer.
            function_name (str): the name of the remote function to call on the peer.
            args (Optional): the args to call with the remote function.
            kwargs (Optional): the kwargs to call with the remote function.

        Returns:
            Future: a future result of the async call.
      )d"))
      .def("async_", &RpcWrapper::async_kwargs)
      .def("async_", &RpcWrapper::async_args)
      .def("async_", &RpcWrapper::async_noargs)
      .def("async_callback", &RpcWrapper::asyncCallback_args_kwargs, py::doc(R"d(
        Make an asynchronous call to ``function_name`` and execute a callback with the result.

        Note:
            If you send a Tensor as an argument, you **may not** modify the Tensor data until the
            result is returned.

        Args:
            name (str): the name of the peer
            function_name (str): the name of the remote function to call on the peer.
            callback (Callable): a callback that accepts the result of the ``Future``.
            args (Optional): the args to call with the remote function.
            kwargs (Optional): the kwargs to call with the remote function.
      )d"))
      .def("async_callback", &RpcWrapper::asyncCallback_kwargs)
      .def("async_callback", &RpcWrapper::asyncCallback_args)
      .def("async_callback", &RpcWrapper::asyncCallback_noargs)
      .def("sync", &RpcWrapper::sync_args_kwargs, py::doc(R"d(
        Make an synchronous call to ``function_name`` and return the result.

        Args:
            peer_name (str): a name.
            function_name (Callable[[], None]): the name of a callback.
            args: the arguments to call with the callback.
            kwargs: the arguments to call with the callback.
      )d"))
      .def("sync", &RpcWrapper::sync_kwargs)
      .def("sync", &RpcWrapper::sync_args)
      .def("sync", &RpcWrapper::sync_noargs)
      .def("set_timeout", &RpcWrapper::setTimeout, py::doc(R"d(
        Set the timeout in secs for each remote call.

        Args:
            timeout (float): the number of seconds before timeout.
      )d"))
      .def("set_transports", &RpcWrapper::setTransports, py::doc(R"d(
        Set which transports are available for the rpc to communicate over.

        Moolib has an internal bandit that automatically selects the transport to use at any
        particular moment. This bandit tries to optimize for latency (and thus indirectly for
        throughput). Peers internally communicate their available transports and
        network addresses, and connections will be attempted with all available transports.

        Args:
            transports (List[str]): a list of transports to use [``"tcp/ip"``, ``"shared memory"``, ``"infiniband"``].
      )d"))
      .def("debug_info", &RpcWrapper::debugInfo, py::doc(R"d(
        Print debugging info.
      )d"));
  py::class_<FutureWrapper, std::shared_ptr<FutureWrapper>>(m, "Future", R"d(A future result.)d")
      .def("result", (py::object(FutureWrapper::*)()) & FutureWrapper::result)
      .def("result", (py::object(FutureWrapper::*)(float)) & FutureWrapper::result, py::arg("timeout"), py::doc(R"d(
        Block on the result of the Future.

        Args:
            timeout (Optional[int]): secs to wait for
      )d"))
      .def("cancel", &FutureWrapper::cancel, py::doc(R"d(Cancel the future calculation.)d"))
      .def("done", &FutureWrapper::done, py::doc(R"d(Check if the future has finished.)d"))
      .def("exception", &FutureWrapper::exception, py::doc("Returns the exception that occurred, if any."))
      .def("__await__", &FutureWrapper::await)
      .def("__iter__", &FutureWrapper::await);
  py::class_<Broker>(m, "Broker", R"d(
    A class to coordinate a cohort during training.

    A broker is a class that should run permanently during your training, and it's primary
    function is to do simple administration for the group. It maintains a list of the peers
    and their addresses, and sends updates when new peers join or disconnect. It uses an
    underlying :class:`Rpc` class to do its communication and this can either be passed in
    or created on construction.

    The broker's listening address is the address your ``Accumulator`` should connect to, and the
    broker should call ``update`` regularly. An example is provided in ``examples/impala/broker.py``
    and below.

    Example:
        A simple broker::

            # In this case create your own Rpc object
            broker_rpc = moolib.Rpc()

            # Pass into broker or construct your own
            broker = moolib.Broker(broker_rpc)

            # Call methods on Rpc object or Broker
            broker_rpc.set_name("broker")
            broker_rpc.listen(flags.address)

            while True:
                broker.update()
                time.sleep(0.25)
  )d")
      .def(
          py::init([](RpcWrapper* rpc) {
            if (rpc)
              return Broker(rpc->rpc);
            else
              return Broker();
          }),
          py::arg("rpc") = py::none(), py::doc(R"d(
        Init.

        The :class:`Broker` uses an underlying :class:`Rpc` object for communication. This
        can be passed in as an argument, or will be constructed by default. As such certain
        calls on the ``Broker`` are simply transparent calls to the underlying method.

        Args:
            rpc (Optional[moolib.Rpc]): An optional :class:`Rpc` object to use.
      )d"))
      .def("set_name", &Broker::setName, py::arg("name"), py::doc(R"d(
        Call `set_name` on the underlying :class:`Rpc` peer.

        see :class:`Rpc` ``set_name``
      )d"))
      .def("listen", &Broker::listen, py::arg("address"), py::doc(R"d(
        Call `listen` on the underlying :class:`Rpc` peer.

        see :class:`Rpc` ``listen``
      )d"))
      .def("update", &Broker::update, py::call_guard<py::gil_scoped_release>(), py::doc(R"d(
        Update the state of the Broker.
      )d"));
  py::class_<GroupWrapper>(m, "Group", R"d(
    A group of Rpc objects.

    This class groups a number of Rpc objects under one unique namespace (the group ``name``),
    to allow for coordinated AllReduce actions.
  )d")
      .def(py::init<RpcWrapper&, std::string>(), py::arg("rpc"), py::arg("name"), py::doc(R"d(
      Init.

      Args:
          rpc (Rpc): an :class:`Rpc` object.
          name (str): the name of the group.
      )d"))
      .def("update", &Group::update, py::call_guard<py::gil_scoped_release>(), py::doc(R"d(
        Update the state of the ``Group``.
      )d"))
      .def("set_broker_name", &Group::setBrokerName)
      .def("set_timeout", &Group::setTimeout)
      .def("set_sort_order", &Group::setSortOrder)
      .def("members", &Group::members)
      .def("sync_id", &Group::syncId)
      .def("name", &Group::name)
      .def("active", &Group::active)
      .def("all_reduce", &GroupWrapper::allReduce);
  py::class_<rpc::Error>(m, "RpcError", R"d(Custom exception for Rpc errors.)d").def("__str__", [](rpc::Error& e) {
    return std::string(e.what());
  });
  py::class_<AllReduceWrapper, std::shared_ptr<AllReduceWrapper>>(
      m, "AllReduce", "A future result of an AllReduce operation.")
      .def("result", (py::object(AllReduceWrapper::*)()) & AllReduceWrapper::result)
      .def(
          "result", (py::object(AllReduceWrapper::*)(float)) & AllReduceWrapper::result, py::arg("timeout"),
          py::doc(R"d(
        Block on the result of the AllReduce Future.

        Args:
            timeout (Optional[int]): secs to wait for.
      )d"))
      .def("cancel", &AllReduceWrapper::cancel, py::doc(R"d(Cancel the future calculation.)d"))
      .def("done", &AllReduceWrapper::done, py::doc(R"d(Check if the future has finished.)d"))
      .def("exception", &AllReduceWrapper::exception, py::doc("Returns the exception that occurred, if any."))
      .def("__await__", &AllReduceWrapper::await)
      .def("__iter__", &AllReduceWrapper::await);
}

} // namespace moolib
