/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "async.h"
#include "function.h"
#include "memory/allocator.h"
#include "memory/buffer.h"
#include "serialization.h"
#include "synchronization.h"

#include <cstddef>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace rpc {

struct Error : std::exception {
  std::string str;
  Error() {}
  Error(std::string&& str) : str(std::move(str)) {}
  virtual const char* what() const noexcept override {
    return str.c_str();
  }
};
extern async::SchedulerFifo scheduler;

template<typename R>
using CallbackFunction = Function<void(R*, Error* error)>;

template<typename T>
struct Me {
  T* me = nullptr;
  Me() = default;
  Me(std::nullptr_t) noexcept {}
  explicit Me(T* me) noexcept : me(me) {
    if (me) {
      me->activeOps.fetch_add(1, std::memory_order_relaxed);
    }
  }
  Me(const Me&) = delete;
  Me(Me&& n) noexcept {
    me = std::exchange(n.me, nullptr);
  }
  Me& operator=(const Me&) = delete;
  Me& operator=(Me&& n) noexcept {
    std::swap(me, n.me);
    return *this;
  }
  ~Me() {
    if (me) {
      me->activeOps.fetch_sub(1);
    }
  }
  T* operator->() const noexcept {
    return me;
  }
  T& operator*() const noexcept {
    return *me;
  }
  explicit operator bool() const noexcept {
    return me;
  }
};

template<typename T>
auto makeMe(T* v) {
  return Me<T>(v);
}

struct TensorContext {
  std::optional<rpc::CUDAStream> stream;
  void synchronize() {
#ifdef USE_CUDA
    if (stream) {
      stream->synchronize();
    }
#endif
  }
};

enum ReqType : uint32_t {
  reqGreeting,
  reqError,
  reqSuccess,
  reqAck,
  reqNack,
  reqFunctionNotFound,
  reqPoke,
  reqLookingForPeer,
  reqPeerFound,
  reqClose,

  reqCallOffset = 1000,
};

template<typename T>
struct RpcDeferredReturn {
  Function<void(BufferHandle)> f;
  template<typename U = T, std::enable_if_t<std::is_same_v<U, void>>* = nullptr>
  void operator()() const {
    f(serializeToBuffer((uint32_t)0, (uint32_t)reqSuccess));
  }
  template<typename U = T, std::enable_if_t<!std::is_same_v<U, void>>* = nullptr>
  void operator()(const U& v) {
    f(serializeToBuffer((uint32_t)0, (uint32_t)reqSuccess, v));
  }
  void operator()(BufferHandle buffer) {
    f(std::move(buffer));
  }
  explicit operator bool() const noexcept {
    return f != nullptr;
  }

  template<typename U = T, std::enable_if_t<std::is_same_v<U, void>>* = nullptr>
  BufferHandle serializeReturnValue() {
    return serializeToBuffer((uint32_t)0, (uint32_t)reqSuccess);
  }
  template<typename U = T, std::enable_if_t<!std::is_same_v<U, void>>* = nullptr>
  BufferHandle serializeReturnValue(const U& v) {
    return serializeToBuffer((uint32_t)0, (uint32_t)reqSuccess, v);
  }
};

struct Rpc {

  using ResponseCallback = Function<void(BufferHandle buffer, TensorContext&, Error*)>;

  Rpc();
  ~Rpc();
  void close();

  struct Service {
    std::string name;
    SpinMutex initMutex;
    std::atomic<void*> pointer = nullptr;
    Function<void()> dtor = nullptr;
    Function<void()> close = nullptr;
    Service(std::string_view name) : name(name) {}
    Service() = default;
    Service(Service&& n) {
      pointer = n.pointer.exchange(nullptr);
      dtor = std::move(n.dtor);
    }
    Service(const Service&) = delete;
    ~Service() {
      if (pointer) {
        std::move(dtor)();
        pointer = nullptr;
      }
    }
    template<typename T, typename... Args>
    T& emplace(Args&&... args) {
      if (!pointer.load(std::memory_order_relaxed)) {
        std::lock_guard l(initMutex);
        if (!pointer) {
          auto* p = new T(std::forward<Args>(args)...);
          dtor = [p]() mutable { delete p; };
          close = [p]() mutable { p->close(); };
          pointer = p;
        }
      }
      return as<T>();
    }
    template<typename T>
    T& as() {
      return *(T*)pointer.load(std::memory_order_relaxed);
    }
    operator std::string_view() const {
      return name;
    }
  };

  template<typename T, typename... Args>
  T* getService(std::string_view name, Args&&... args) {
    std::unique_lock l(servicesMutex);
    auto& s = const_cast<Service&>(*services[typeid(T)].emplace(name).first);
    l.unlock();
    return &s.emplace<T>(*this, std::forward<Args>(args)...);
  }

  void setName(std::string_view name);
  std::string_view getName() const;
  void setOnError(Function<void(const Error&)>&&);
  void listen(std::string_view addr);
  void connect(std::string_view addr);

  enum class ExceptionMode { None, DeserializationOnly, All };

  void setExceptionMode(ExceptionMode mode) {
    currentExceptionMode_ = mode;
  }

  struct FBase {
    virtual ~FBase(){};
    virtual void call(BufferHandle, TensorContext&, Function<void(BufferHandle)>) = 0;
  };

  template<typename Signature, typename F>
  struct FImpl;

  template<typename R, typename... Args, typename F>
  struct FImpl<R(Args...), F> : FBase {
    Rpc& rpc;
    F f;
    template<typename F2>
    FImpl(Rpc& rpc, F2&& f) : rpc(rpc), f(std::forward<F2>(f)) {}
    virtual ~FImpl() {}
    virtual void
    call(BufferHandle inbuffer, TensorContext& tensorContext, Function<void(BufferHandle)> callback) noexcept override {
      try {
        std::tuple<std::decay_t<Args>...> args;
        constexpr bool takesStream =
            std::is_invocable_r_v<R, F, std::optional<CUDAStream>, Args...> ||
            std::is_invocable_r_v<void, F, RpcDeferredReturn<R>, std::optional<CUDAStream>, Args...>;
        constexpr bool isDeferred =
            std::is_invocable_r_v<void, F, RpcDeferredReturn<R>, Args...> ||
            std::is_invocable_r_v<void, F, RpcDeferredReturn<R>, std::optional<CUDAStream>, Args...>;
        constexpr bool takesBuffer = [] {
          if constexpr (sizeof...(Args) == 1) {
            return std::is_same_v<std::decay_t<std::tuple_element_t<0, std::tuple<Args...>>>, BufferHandle>;
          } else {
            return false;
          }
        }();
        auto in = [&]() {
          uint32_t rid, fid;
          if constexpr (takesBuffer) {
            std::get<0>(args) = std::move(inbuffer);
          } else {
            std::apply(
                [&](std::decay_t<Args>&... args) { deserializeBuffer(std::move(inbuffer), rid, fid, args...); }, args);
          }
        };
        BufferHandle outbuffer;
        auto out = [this, &args, &outbuffer, &tensorContext]() {
          if constexpr (!isDeferred) {
            auto wrap = [&](auto&&... args) {
              if constexpr (takesStream) {
                return f(tensorContext.stream, std::forward<decltype(args)>(args)...);
              } else {
                if (tensorContext.stream) {
                  tensorContext.stream->synchronize();
                }
                return f(std::forward<decltype(args)>(args)...);
              }
            };
            if constexpr (std::is_same_v<void, R>) {
              std::apply(wrap, std::move(args));
              serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqSuccess);
            } else if constexpr (std::is_same_v<R, BufferHandle>) {
              outbuffer = std::apply(wrap, std::move(args));
            } else {
              serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqSuccess, std::apply(wrap, std::move(args)));
            }
          }
        };
        auto exceptionMode = rpc.currentExceptionMode_.load(std::memory_order_relaxed);
        if (exceptionMode == ExceptionMode::None) {
          in();
          out();
        } else if (exceptionMode == ExceptionMode::DeserializationOnly) {
          try {
            in();
          } catch (const std::exception& e) {
            serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqError, std::string_view(e.what()));
            std::move(callback)(std::move(outbuffer));
            return;
          }
          out();
        } else {
          try {
            in();
            out();
          } catch (const std::exception& e) {
            serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqError, std::string_view(e.what()));
            std::move(callback)(std::move(outbuffer));
            return;
          }
        }
        if constexpr (isDeferred) {
          RpcDeferredReturn<R> d;
          d.f = std::move(callback);
          auto wrap = [&](auto&&... args) {
            if constexpr (takesStream) {
              f(std::move(d), tensorContext.stream, std::forward<decltype(args)>(args)...);
            } else {
              if (tensorContext.stream) {
                tensorContext.stream->synchronize();
              }
              f(std::move(d), std::forward<decltype(args)>(args)...);
            }
          };
          std::apply(wrap, std::move(args));
        } else {
          std::move(callback)(std::move(outbuffer));
        }
      } catch (const std::exception& e) {
        fprintf(stderr, "Unhandled exception in RPC function: %s\n", e.what());
        std::abort();
      }
    }
  };

  template<typename Signature, typename F>
  void define(std::string_view name, F&& f) {
    auto ff = std::make_unique<FImpl<Signature, F>>(*this, std::forward<F>(f));
    define(name, std::move(ff));
  }
  void undefine(std::string_view name);

  template<typename... Args>
  static BufferHandle serializeArguments(const Args&... args) {
    BufferHandle buffer;
    serializeToBuffer(buffer, (uint32_t)0, (uint32_t)0, args...);
    return buffer;
  }

  template<typename... Args>
  static void serializeReturn(BufferHandle& buffer, const Args&... args) {
    serializeToBuffer(buffer, (uint32_t)0, (uint32_t)reqSuccess, args...);
  }

  template<typename... Args>
  static BufferHandle serializeReturn(const Args&... args) {
    BufferHandle buffer;
    serializeReturn(buffer, args...);
    return buffer;
  }

  template<typename Buffer, typename... Args>
  static void deserializeArguments(Buffer& buffer, Args&... args) {
    uint32_t rid, fid;
    deserializeBuffer(buffer, rid, fid, args...);
  }

  template<typename R, typename Callback, typename... Args>
  void
  asyncCallback(std::string_view peerName, std::string_view functionName, Callback&& callback, const Args&... args) {
    BufferHandle buffer;
    if constexpr (sizeof...(args) == 1) {
      if constexpr (std::is_same_v<Args..., BufferHandle>) {
        buffer = std::move(const_cast<BufferHandle&>(args)...);
      } else {
        serializeToBuffer(buffer, (uint32_t)0, (uint32_t)0, args...);
      }
    } else {
      serializeToBuffer(buffer, (uint32_t)0, (uint32_t)0, args...);
    }

    sendRequest(
        peerName, functionName, std::move(buffer),
        [this, callback = std::forward<Callback>(callback)](
            BufferHandle buffer, TensorContext& tensorContext, Error* error) mutable noexcept {
          constexpr bool takesStream = std::is_invocable_v<Callback, std::optional<rpc::CUDAStream>, R*, Error*>;
          constexpr bool takesReturnValue = takesStream || std::is_invocable_v<Callback, R*, Error*>;
          constexpr bool takesError = std::is_invocable_v<Callback, Error*>;
          static_assert(
              takesReturnValue || takesError || std::is_same_v<std::decay_t<Callback>, std::nullptr_t>,
              "Callback function has invalid signature");
          static constexpr auto wrap = [](auto&& f, std::optional<CUDAStream> stream, R* r, rpc::Error* error) {
            if constexpr (takesStream) {
              return std::forward<decltype(f)>(f)(stream, r, error);
            } else {
              if (stream) {
                stream->synchronize();
              }
              return std::forward<decltype(f)>(f)(r, error);
            }
          };
          if constexpr (takesReturnValue) {
            if (error) {
              scheduler.run([me = makeMe(this), callback = std::move(callback), error = std::move(*error)]() mutable {
                std::optional<CUDAStream> nullStream;
                wrap(std::move(callback), nullStream, nullptr, &error);
              });
            } else {
              scheduler.run([me = makeMe(this), callback = std::move(callback), buffer = std::move(buffer),
                             tensorContext = std::move(tensorContext)]() mutable {
                std::optional<CUDAStreamGuard> streamGuard;
                if (tensorContext.stream) {
                  streamGuard.emplace(*tensorContext.stream);
                }
                if constexpr (std::is_same_v<R, void>) {
                  char nonnull;
                  wrap(std::move(callback), tensorContext.stream, (void*)&nonnull, nullptr);
                } else {
                  uint32_t rid, fid;
                  std::optional<R> r;
                  try {
                    r.emplace();
                    deserializeBuffer(buffer, rid, fid, *r);
                  } catch (const std::exception& e) {
                    Error err{std::string("Deserialization error: ") + e.what()};
                    wrap(std::move(callback), tensorContext.stream, nullptr, &err);
                    return;
                  }
                  wrap(std::move(callback), tensorContext.stream, &*r, nullptr);
                }
              });
            }
          } else if constexpr (takesError) {
            if (error) {
              scheduler.run([me = makeMe(this), callback = std::move(callback), error = std::move(*error)]() mutable {
                std::move(callback)(&error);
              });
            }
          }
        });
  }

  template<typename R = void, typename... Args>
  std::future<R> async(std::string_view peerName, std::string_view functionName, const Args&... args) {
    std::promise<R> promise;
    auto future = promise.get_future();
    asyncCallback<R>(
        peerName, functionName,
        [promise = std::move(promise)]([[maybe_unused]] R* ptr, Error* err) mutable {
          if (err) {
            promise.set_exception(std::make_exception_ptr(std::move(*err)));
          } else {
            if constexpr (std::is_same_v<R, void>) {
              promise.set_value();
            } else {
              promise.set_value(std::move(*ptr));
            }
          }
        },
        args...);
    return future;
  }

  template<typename R = void, typename... Args>
  R sync(std::string_view peerName, std::string_view functionName, const Args&... args) {
    auto future = async<R>(peerName, functionName, args...);
    return future.get();
  }

  void setTimeout(std::chrono::milliseconds milliseconds);
  std::chrono::milliseconds getTimeout();

  void debugInfo();

  void setTransports(const std::vector<std::string>& names);

  struct Impl;

private:
  friend Me<Rpc>;
  void
  sendRequest(std::string_view peerName, std::string_view functionName, BufferHandle buffer, ResponseCallback response);

  std::atomic_int activeOps{0};
  std::atomic<ExceptionMode> currentExceptionMode_ = ExceptionMode::DeserializationOnly;
  std::unique_ptr<Impl> impl_;

  SpinMutex servicesMutex;
  std::unordered_map<
      std::type_index, std::unordered_set<Service, std::hash<std::string_view>, std::equal_to<std::string_view>>>
      services;

  void define(std::string_view name, std::unique_ptr<FBase>&& f);
};

} // namespace rpc
