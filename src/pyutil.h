/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "pybind11/pybind11.h"

#include <mutex>

namespace moolib {

namespace py = pybind11;

template<typename T>
struct glock {
  std::unique_lock<T> lock;
  glock(T& mutex) : lock(mutex, std::try_to_lock) {
    if (!lock.owns_lock()) {
      if (PyGILState_Check()) {
        py::gil_scoped_release gil;
        lock.lock();
      } else {
        lock.lock();
      }
    }
  }
};

template<typename T>
struct GilWrapper {
  std::optional<T> obj;
  GilWrapper() = default;
  GilWrapper(const T& n) {
    py::gil_scoped_acquire gil;
    obj = n;
  }
  GilWrapper(T&& n) {
    obj = std::move(n);
  }
  GilWrapper(const GilWrapper& n) {
    py::gil_scoped_acquire gil;
    obj = n.obj;
  }
  GilWrapper(GilWrapper&& n) {
    obj = std::move(n.obj);
  }
  ~GilWrapper() {
    if (obj && *obj) {
      py::gil_scoped_acquire gil;
      obj.reset();
    }
  }
  T release() {
    T r = std::move(obj.value());
    obj.reset();
    return r;
  }
  void reset() {
    obj.reset();
  }
  operator bool() const noexcept {
    return obj.has_value();
  }
  T& operator*() & {
    return *obj;
  }
  T&& operator*() && {
    return std::move(*obj);
  }
  T* operator->() {
    return &*obj;
  }
  GilWrapper& operator=(const GilWrapper& n) {
    py::gil_scoped_acquire gil;
    obj = n.obj;
    return *this;
  }
  GilWrapper& operator=(GilWrapper&& n) {
    if (obj && *obj) {
      // acquire GIL here as existing object needs to be released
      py::gil_scoped_acquire gil;
      obj = std::move(n.obj);
    } else {
      obj = std::move(n.obj);
    }
    return *this;
  }
  template<typename X>
  void serialize(X& x) {
    py::gil_scoped_acquire gil;
    obj.emplace();
    x(*obj);
  }
  template<typename X>
  void serialize(X& x) const {
    py::gil_scoped_acquire gil;
    x(*obj);
  }
};

} // namespace moolib
