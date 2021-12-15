/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "allocator.h"
#include "async.h"
#include "buffer.h"
#include "function.h"
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

template<typename T>
struct RpcDeferredReturn {
  rpc::Function<std::conditional_t<std::is_same_v<void, T>, void(), void(const T&)>> f;
  template<typename U>
  void operator()(U&& u) {
    f(std::forward<U>(u));
  }
};

struct Rpc {

  using ResponseCallback = Function<void(BufferHandle buffer, Error*)>;

  Rpc();
  ~Rpc();
  void close();

  struct Service {
    std::string name;
    void* pointer = nullptr;
    Function<void()> dtor = nullptr;
    Service(std::string_view name) : name(name) {}
    Service() = default;
    Service(Service&& n) {
      pointer = std::exchange(n.pointer, nullptr);
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
      if (!pointer) {
        pointer = new T(std::forward<Args>(args)...);
        dtor = [pointer = pointer]() { delete (T*)pointer; };
      }
      return as<T>();
    }
    template<typename T>
    T& as() {
      return *(T*)pointer;
    }
    operator std::string_view() const {
      return name;
    }
  };

  template<typename T>
  T* getService(std::string_view name) {
    return &const_cast<Service&>(*services[typeid(T)].emplace(name).first).emplace<T>(*this);
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
    virtual void call(BufferHandle, Function<void(BufferHandle)>) = 0;
  };

  template<typename Signature, typename F>
  struct FImpl;

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

  template<typename R, typename... Args, typename F>
  struct FImpl<R(Args...), F> : FBase {
    Rpc& rpc;
    F f;
    template<typename F2>
    FImpl(Rpc& rpc, F2&& f) : rpc(rpc), f(std::forward<F2>(f)) {}
    virtual ~FImpl() {}
    virtual void call(BufferHandle inbuffer, Function<void(BufferHandle)> callback) noexcept override {
      scheduler.run([rpc = makeMe(&rpc), this, inbuffer = std::move(inbuffer),
                     callback = std::move(callback)]() mutable noexcept {
        try {
          std::tuple<std::decay_t<Args>...> args;
          constexpr bool isDeferred = std::is_invocable_r_v<void, F, RpcDeferredReturn<R>, Args...>;
          auto in = [&]() {
            uint32_t rid, fid;
            std::apply(
                [&](std::decay_t<Args>&... args) { deserializeBuffer(std::move(inbuffer), rid, fid, args...); }, args);
          };
          BufferHandle outbuffer;
          auto out = [this, &args, &outbuffer]() {
            if constexpr (!isDeferred) {
              if constexpr (std::is_same_v<void, R>) {
                std::apply(f, std::move(args));
                serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqSuccess);
              } else {
                serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqSuccess, std::apply(f, std::move(args)));
              }
            }
          };
          auto exceptionMode = rpc->currentExceptionMode_.load(std::memory_order_relaxed);
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
            if constexpr (std::is_same_v<void, R>) {
              d.f = [callback = std::move(callback)]() mutable noexcept {
                BufferHandle outbuffer;
                serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqSuccess);
                std::move(callback)(std::move(outbuffer));
              };
            } else {
              d.f = [callback = std::move(callback)](const R& r) mutable noexcept {
                BufferHandle outbuffer;
                serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqSuccess, r);
                std::move(callback)(std::move(outbuffer));
              };
            }
            auto wrap = [&](auto&&... args) { f(std::move(d), std::forward<decltype(args)>(args)...); };
            std::apply(wrap, std::move(args));
          } else {
            std::move(callback)(std::move(outbuffer));
          }
        } catch (const std::exception& e) {
          fprintf(stderr, "Unhandled exception in RPC function: %s\n", e.what());
          std::abort();
        }
      });
    }
  };

  template<typename Signature, typename F>
  void define(std::string_view name, F&& f) {
    auto ff = std::make_unique<FImpl<Signature, F>>(*this, std::forward<F>(f));
    define(name, std::move(ff));
  }

  template<typename... Args>
  BufferHandle serializeArguments(const Args&... args) {
    BufferHandle buffer;
    serializeToBuffer(buffer, (uint32_t)0, (uint32_t)0, args...);
    return buffer;
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
        [this, callback = std::forward<Callback>(callback)](BufferHandle buffer, Error* error) mutable noexcept {
          constexpr bool takesReturnValue = std::is_invocable_v<Callback, R*, Error*>;
          constexpr bool takesError = std::is_invocable_v<Callback, Error*>;
          static_assert(
              takesReturnValue || takesError || std::is_same_v<std::decay_t<Callback>, std::nullptr_t>,
              "Callback function has invalid signature");
          if constexpr (takesReturnValue) {
            if (error) {
              scheduler.run([me = makeMe(this), callback = std::move(callback), error = std::move(*error)]() mutable {
                std::move(callback)(nullptr, &error);
              });
            } else {
              scheduler.run([me = makeMe(this), callback = std::move(callback), buffer = std::move(buffer)]() mutable {
                if constexpr (std::is_same_v<R, void>) {
                  char nonnull;
                  std::move(callback)((void*)&nonnull, nullptr);
                } else {
                  uint32_t rid, fid;
                  std::optional<R> r;
                  try {
                    r.emplace();
                    deserializeBuffer(buffer, rid, fid, *r);
                  } catch (const std::exception& e) {
                    Error err{std::string("Deserialization error: ") + e.what()};
                    std::move(callback)(nullptr, &err);
                    return;
                  }
                  std::move(callback)(&*r, nullptr);
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
  std::unordered_map<
      std::type_index, std::unordered_set<Service, std::hash<std::string_view>, std::equal_to<std::string_view>>>
      services;

  std::atomic<ExceptionMode> currentExceptionMode_ = ExceptionMode::DeserializationOnly;
  std::unique_ptr<Impl> impl_;

  void define(std::string_view name, std::unique_ptr<FBase>&& f);
};

} // namespace rpc
