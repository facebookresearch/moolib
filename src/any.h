/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <type_traits>

namespace moolib {

template<size_t embeddedSize>
struct Any {
  std::aligned_storage_t<std::max(embeddedSize, sizeof(void*)), alignof(std::max_align_t)> buf;
  void (*dtor)(Any*) = nullptr;
  template<typename T>
  constexpr bool embed() const noexcept {
    return sizeof(T) <= embeddedSize;
  }
  Any() = default;
  Any(const Any&) = delete;
  Any(Any&&) = delete;
  ~Any() {
    if (dtor) {
      dtor(this);
    }
  }
  Any& operator=(const Any&) = delete;
  Any& operator=(Any&&) = delete;
  template<typename T>
  T* pointer() const noexcept {
    return embed<T>() ? (T*)&buf : (T*&)buf;
  }
  template<typename T>
  T& as() noexcept {
    return *pointer<T>();
  }
  template<typename T>
  const T& as() const noexcept {
    return *pointer<T>();
  }
  template<typename T, typename... Args>
  T& emplace(Args&&... args) {
    if (dtor) {
      dtor(this);
    }
    T* p;
    if (embed<T>()) {
      p = pointer<T>();
      new (p) T(std::forward<Args>(args)...);
      dtor = [](Any* me) {
        me->as<T>().~T();
        me->dtor = nullptr;
      };
    } else {
      p = new T(std::forward<Args>(args)...);
      (T*&)buf = p;
      dtor = [](Any* me) {
        delete me->pointer<T>();
        me->dtor = nullptr;
      };
    }
    return *p;
  }
};

} // namespace moolib
