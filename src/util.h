/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "logging.h"
#include "rpc.h"
#include "tensor.h"

#include <chrono>
#include <random>
#include <thread>

namespace moolib {

inline auto seedRng() {
  std::random_device dev;
  auto start = std::chrono::high_resolution_clock::now();
  std::seed_seq ss(
      {(uint32_t)dev(), (uint32_t)dev(), (uint32_t)(std::chrono::high_resolution_clock::now() - start).count(),
       (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count(), (uint32_t)dev(),
       (uint32_t)std::chrono::system_clock::now().time_since_epoch().count(),
       (uint32_t)std::chrono::high_resolution_clock::now().time_since_epoch().count(),
       (uint32_t)(std::chrono::high_resolution_clock::now() - start).count(), (uint32_t)dev(),
       (uint32_t)(std::chrono::high_resolution_clock::now() - start).count(), (uint32_t)dev(),
       (uint32_t)std::hash<std::thread::id>()(std::this_thread::get_id())});
  return std::mt19937_64(ss);
};

inline std::mt19937_64& getRng() {
  thread_local std::mt19937_64 rng{seedRng()};
  return rng;
}

template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
T random(T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max()) {
  return std::uniform_int_distribution<T>(min, max)(getRng());
}

template<typename Duration>
float seconds(Duration duration) {
  return std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(duration).count();
}

struct Timer {
  std::chrono::steady_clock::time_point start;
  Timer() {
    reset();
  }
  void reset() {
    start = std::chrono::steady_clock::now();
  }
  float elapsedAt(std::chrono::steady_clock::time_point now) {
    return std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(now - start).count();
  }
  float elapsed() {
    return elapsedAt(std::chrono::steady_clock::now());
  }
  float elapsedReset() {
    auto now = std::chrono::steady_clock::now();
    float r = elapsedAt(now);
    start = now;
    return r;
  }
};

inline std::string randomName() {
  std::string r;
  for (int i = 0; i != 16; ++i) {
    r += "0123456789abcdef"[std::uniform_int_distribution<int>(0, 15)(getRng())];
  }
  return r;
}

inline int getTensorDType(char dtype, int itemsize) {
  using rpc::Tensor;
  switch (dtype) {
  case 'f':
    if (itemsize == 2) {
      return Tensor::kFloat16;
    } else if (itemsize == 4) {
      return Tensor::kFloat32;
    } else if (itemsize == 8) {
      return Tensor::kFloat64;
    } else {
      throw std::runtime_error("Unexpected itemsize for float");
    }
    break;
  case 'i':
    if (itemsize == 1) {
      return Tensor::kInt8;
    } else if (itemsize == 2) {
      return Tensor::kInt16;
    } else if (itemsize == 4) {
      return Tensor::kInt32;
    } else if (itemsize == 8) {
      return Tensor::kInt64;
    } else
      throw std::runtime_error("Unexpected itemsize for int");
    break;
  case 'u':
    if (itemsize == 1) {
      return Tensor::kUInt8;
    } else
      throw std::runtime_error("Unexpected itemsize for unsigned int");
    break;
  case 'b':
    if (itemsize == 1) {
      return Tensor::kBool;
    } else
      throw std::runtime_error("Unexpected itemsize for boolean");
    break;
  default:
    throw std::runtime_error("Unsupported dtype '" + std::string(1, dtype) + "'");
  }
}

template<typename T>
struct Future {
private:
  using IT = std::conditional_t<std::is_same_v<T, void>, std::nullptr_t, T>;
  struct S {
    std::optional<IT> value;
    std::atomic_bool hasValue = false;
  };
  std::shared_ptr<S> s;

public:
  Future() {
    s = std::make_shared<S>();
  }
  void reset() {
    *this = Future();
  }
  void set() {
    s->value.emplace();
    s->hasValue = true;
  }
  template<typename T2>
  void set(T2&& val) {
    s->value = std::move(val);
    s->hasValue = true;
  }
  operator bool() const noexcept {
    return s->hasValue;
  }
  IT& operator*() {
    return *s->value;
  }
  IT* operator->() {
    return &*s->value;
  }
};

template<typename T, typename... Args>
Future<T> callImpl(rpc::Rpc& rpc, std::string_view peerName, std::string_view funcName, Args&&... args) {
  Future<T> retval;
  rpc.asyncCallback<T>(
      peerName, funcName,
      [retval](T* value, rpc::Error* err) mutable {
        if (value) {
          if constexpr (!std::is_same_v<T, void>) {
            retval.set(*value);
          } else {
            retval.set();
          }
        } else {
          log.error("RPC error: %s\n", err->what());
        }
      },
      std::forward<Args>(args)...);
  return retval;
}

template<typename T>
std::string sizesStr(T&& sizes) {
  std::string s = "{";
  for (auto& v : sizes) {
    if (s.size() > 1) {
      s += ", ";
    }
    s += std::to_string(v);
  }
  s += "}";
  return s;
}

template<typename T>
struct Dtor {
  T f;
  Dtor(T f) : f(std::move(f)) {}
  ~Dtor() {
    f();
  }
};

} // namespace moolib
