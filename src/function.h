/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#pragma once

#include <atomic>
#include <cstddef>
#include <cstring>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace rpc {

namespace function {

namespace impl {

struct OpsBase {
  const size_t size = 0;
};

struct alignas(std::max_align_t) Storage {
  union {
    Storage* next = nullptr;
    std::atomic<Storage*> nextAtomic;
    static_assert(std::is_standard_layout_v<decltype(nextAtomic)>);
    static_assert(std::is_trivially_destructible_v<decltype(nextAtomic)>);
  };
  size_t allocated_ = 0;
  const OpsBase* ops_ = nullptr;
  template<typename F>
  F& as() const noexcept {
    return *((F*)(void*)(this + 1));
  }
};

template<typename T>
struct ArgType {
  using type = typename std::conditional_t<std::is_reference_v<T>, T, T&&>;
};

template<typename R, typename... Args>
struct Ops : OpsBase {
  R (*call)(Storage&, Args&&...) = nullptr;
  R (*callAndDtor)(Storage&, Args&&...) = nullptr;
  void (*copyCtor)(Storage&, Storage&) = nullptr;
  void (*copy)(Storage&, Storage&) = nullptr;
  void (*dtor)(Storage&) = nullptr;
};

template<typename F, typename R, typename... Args>
struct OpsConstructor {
  static constexpr Ops<R, Args...> make() {
    Ops<R, Args...> r{{sizeof(F)}};
    r.call = [](Storage& s, Args&&... args) {
      // return std::invoke(s.as<F>(), std::forward<Args>(args)...);
      return s.template as<F>()(std::forward<Args>(args)...);
    };
    r.callAndDtor = [](Storage& s, Args&&... args) {
      F& f = s.template as<F>();
      if constexpr (std::is_same_v<R, void>) {
        f(std::forward<Args>(args)...);
        f.~F();
      } else {
        auto r = f(std::forward<Args>(args)...);
        f.~F();
        return r;
      }
    };
    if (!std::is_trivially_destructible<F>::value) {
      r.dtor = [](Storage& s) { s.template as<F>().~F(); };
      r.copy = [](Storage& to, Storage& from) {
        if constexpr (std::is_copy_assignable_v<F>) {
          to.template as<F>() = from.template as<F>();
        } else {
          throw std::runtime_error("function is not copy assignable");
        }
      };
      r.copyCtor = [](Storage& to, Storage& from) {
        if constexpr (std::is_copy_constructible_v<F>) {
          new (&to.template as<F>()) F(from.template as<F>());
        } else {
          throw std::runtime_error("function is not copy constructible");
        }
      };
    }
    return r;
  }
  static constexpr Ops<R, Args...> value = make();
};
template<typename R, typename... Args>
struct NullOps {
  static constexpr Ops<R, Args...> value{};
};

template<typename T>
struct FreeList {
  T* ptr = nullptr;
  bool dead = false;
  ~FreeList() {
    dead = true;
    while (ptr) {
      T* next = ptr->next;
      std::free(ptr);
      ptr = next;
    }
    ptr = nullptr;
  }

  static FreeList& get() {
    thread_local FreeList freeList;
    return freeList;
  }
};

inline Storage* allocStorage(size_t n) {
  void* ptr = std::malloc(sizeof(Storage) + n);
  if (!ptr) {
    throw std::bad_alloc();
  }
  Storage* r = new (ptr) Storage();
  r->allocated_ = n;
  r->ops_ = &NullOps<void>::value;
  return r;
}

inline Storage* newStorage(size_t n) {
  auto& fl = FreeList<Storage>::get();
  if (!fl.ptr) {
    return allocStorage(n);
  }
  return std::exchange(fl.ptr, fl.ptr->next);
}

inline void freeStorage(Storage* s) {
  auto& fl = FreeList<Storage>::get();
  if (fl.dead) {
    std::free(s);
    return;
  }
  s->next = fl.ptr;
  fl.ptr = s;
}

inline void getStorage(Storage*& s, size_t n) {
  if (!s) {
    s = newStorage(n + 32);
  }
  if (s->allocated_ < n) {
    std::free(s);
    try {
      s = allocStorage(n + 32);
    } catch (...) {
      s = nullptr;
      throw;
    }
  }
}

} // namespace impl

using FunctionPointer = impl::Storage*;

template<typename T>
class Function;
template<typename R, typename... Args>
class Function<R(Args...)> {
  impl::Storage* storage_ = nullptr;
  const impl::Ops<R, Args...>* ops_ = &impl::NullOps<R, Args...>::value;

public:
  Function() = default;
  Function(const Function& n) {
    *this = n;
  }
  Function(Function&& n) noexcept {
    *this = std::move(n);
  }
  Function(std::nullptr_t) noexcept {}
  Function(FunctionPointer ptr) noexcept {
    storage_ = ptr;
    ops_ = (const impl::Ops<R, Args...>*)ptr->ops_;
  }

  template<typename F>
  Function(F&& f) {
    *this = std::forward<F>(f);
  }

  ~Function() {
    *this = nullptr;
  }

  FunctionPointer release() noexcept {
    auto r = storage_;
    storage_ = nullptr;
    ops_ = &impl::NullOps<R, Args...>::value;
    return r;
  }

  Function& operator=(FunctionPointer ptr) noexcept {
    *this = nullptr;
    storage_ = ptr;
    ops_ = (const impl::Ops<R, Args...>*)ptr->ops_;
  }

  R operator()(Args... args) const& {
    return ops_->call(*storage_, std::forward<Args>(args)...);
  }

  R operator()(Args... args) && {
    if constexpr (std::is_same_v<R, void>) {
      ops_->callAndDtor(*storage_, std::forward<Args>(args)...);
      ops_ = &impl::NullOps<R, Args...>::value;
      *this = nullptr;
    } else {
      auto r = ops_->callAndDtor(*storage_, std::forward<Args>(args)...);
      ops_ = &impl::NullOps<R, Args...>::value;
      *this = nullptr;
      return r;
    }
  }

  Function& operator=(const Function& n) {
    if (!n.storage_) {
      *this = nullptr;
    } else {
      if (ops_ == n.ops_) {
        if (!ops_->dtor) {
          std::memcpy(&storage_->template as<char>(), &n.storage_->template as<char>(), ops_->size);
        } else {
          ops_->copy(*storage_, *n.storage_);
        }
      } else {
        if (ops_->dtor) {
          ops_->dtor(*storage_);
        }
        try {
          impl::getStorage(storage_, n.ops_->size);
          ops_ = n.ops_;
          storage_->ops_ = ops_;
          if (!ops_->dtor) {
            std::memcpy(&storage_->template as<char>(), &n.storage_->template as<char>(), ops_->size);
          } else {
            ops_->copyCtor(*storage_, *n.storage_);
          }

        } catch (...) {
          ops_ = &impl::NullOps<R, Args...>::value;
          *this = nullptr;
          throw;
        }
      }
    }
    return *this;
  }
  Function& operator=(Function&& n) noexcept {
    std::swap(ops_, n.ops_);
    std::swap(storage_, n.storage_);
    return *this;
  }

  Function& operator=(std::nullptr_t) noexcept {
    if (ops_->dtor) {
      ops_->dtor(*storage_);
      ops_ = &impl::NullOps<R, Args...>::value;
    }
    if (storage_) {
      impl::freeStorage(storage_);
      storage_ = nullptr;
    }
    return *this;
  }

  template<typename F, std::enable_if_t<!std::is_same_v<std::remove_reference_t<F>, Function>>* = nullptr>
  Function& operator=(F&& f) {
    if (ops_->dtor) {
      ops_->dtor(*storage_);
    }
    using FT = std::remove_reference_t<F>;
    static_assert(alignof(F) <= alignof(impl::Storage));
    try {
      impl::getStorage(storage_, sizeof(FT));
      new (&storage_->template as<FT>()) FT(std::forward<F>(f));
    } catch (...) {
      ops_ = &impl::NullOps<R, Args...>::value;
      *this = nullptr;
      throw;
    }
    ops_ = &impl::OpsConstructor<FT, R, Args...>::value;
    storage_->ops_ = ops_;
    return *this;
  }

  explicit operator bool() const noexcept {
    return storage_ != nullptr;
  }

  template<typename T>
  T& as() const noexcept {
    return storage_->as<T>();
  }

  bool operator==(FunctionPointer f) const noexcept {
    return storage_ == f;
  }
  bool operator!=(FunctionPointer f) const noexcept {
    return storage_ != f;
  }
};

} // namespace function

using FunctionPointer = function::FunctionPointer;
template<typename T>
using Function = function::Function<T>;

} // namespace rpc
