#pragma once

#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

namespace moolib {

template<typename T>
struct Vector {
  T* storagebegin = nullptr;
  T* storageend = nullptr;
  T* beginptr = nullptr;
  T* endptr = nullptr;
  size_t msize = 0;
  Vector() = default;
  Vector(const Vector&) = delete;
  Vector(Vector&& n) {
    *this = std::move(n);
  }
  ~Vector() {
    if (beginptr != endptr) {
      clear();
    }
    if (storagebegin) {
      std::free(storagebegin);
    }
  }
  Vector& operator=(const Vector&) = delete;
  Vector& operator=(Vector&& n) {
    std::swap(storagebegin, n.storagebegin);
    std::swap(storageend, n.storageend);
    std::swap(beginptr, n.beginptr);
    std::swap(endptr, n.endptr);
    std::swap(msize, n.msize);
    return *this;
  }
  size_t size() {
    return msize;
  }
  T* data() {
    return beginptr;
  }
  T* begin() {
    return beginptr;
  }
  T* end() {
    return endptr;
  }
  T& operator[](size_t index) {
    return beginptr[index];
  }
  void clear() {
    for (auto* i = beginptr; i != endptr; ++i) {
      i->~T();
    }
    beginptr = storagebegin;
    endptr = beginptr;
    msize = 0;
  }
  void move(T* dst, T* begin, T* end) {
    if constexpr (std::is_trivially_copyable_v<T>) {
      std::memmove((void*)dst, (void*)begin, (end - begin) * sizeof(T));
    } else {
      if (dst <= begin) {
        for (auto* i = begin; i != end;) {
          *dst = std::move(*i);
          ++dst;
          ++i;
        }
      } else {
        auto* dsti = dst + (end - begin);
        for (auto* i = end; i != begin;) {
          --dsti;
          --i;
          *dsti = std::move(*i);
        }
      }
    }
  }
  void erase(T* begin, T* end) {
    for (auto* i = begin; i != end; ++i) {
      i->~T();
    }
    size_t n = end - begin;
    msize -= n;
    if (begin == beginptr) {
      for (auto* i = begin; i != end; ++i) {
        i->~T();
      }
      beginptr = end;
      if (beginptr != endptr) {
        size_t unused = beginptr - storagebegin;
        if (unused > msize && unused >= 1024 * 512 / sizeof(T)) {
          if constexpr (std::is_trivially_copyable_v<T>) {
            move(storagebegin, beginptr, endptr);
          } else {
            auto* sbi = storagebegin;
            auto* bi = beginptr;
            while (sbi != beginptr && bi != endptr) {
              new (sbi) T(std::move(*bi));
              ++sbi;
              ++bi;
            }
            move(sbi, bi, endptr);
            for (auto* i = bi; i != endptr; ++i) {
              i->~T();
            }
          }
          beginptr = storagebegin;
          endptr = beginptr + msize;
        }
      }
    } else {
      move(begin, end, endptr);
      for (auto* i = end; i != endptr; ++i) {
        i->~T();
      }
      endptr -= n;
    }
    if (beginptr == endptr) {
      beginptr = storagebegin;
      endptr = beginptr;
    }
  }
  void resize(size_t n) {
    if (msize > n) {
      T* i = endptr;
      T* e = beginptr + n;
      while (i != e) {
        --i;
        i->~T();
      }
    } else if (n > msize) {
      reserve(n);
      T* i = endptr;
      T* e = beginptr + n;
      while (i != e) {
        new (i) T();
        ++i;
      }
    }
    endptr = beginptr + n;
    msize = n;
  }
  bool empty() const {
    return beginptr == endptr;
  }
  size_t capacity() {
    return storageend - beginptr;
  }
  void reserveImpl(size_t n) {
    auto* lbegin = beginptr;
    auto* lend = endptr;
    auto* prevstorage = storagebegin;
    size_t msize = this->msize;
    T* newptr = (T*)std::aligned_alloc(alignof(T), sizeof(T) * n);
    if (!newptr) {
      throw std::bad_alloc();
    }
    if (prevstorage) {
      if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(newptr, lbegin, sizeof(T) * msize);
      } else {
        T* dst = newptr;
        for (T* i = lbegin; i != lend; ++i) {
          new (dst) T(std::move(*i));
          i->~T();
          ++dst;
        }
      }
      std::free(prevstorage);
    }
    storagebegin = newptr;
    storageend = newptr + n;
    beginptr = newptr;
    endptr = newptr + msize;
  }
  void reserve(size_t n) {
    if (n <= capacity()) {
      return;
    }
    reserveImpl(n);
  }
  void expand() {
    reserveImpl(std::max(capacity() * 2, (size_t)16));
  }
  void push_back(const T& v) {
    emplace_back(v);
  }
  void push_back(T&& v) {
    emplace_back(std::move(v));
  }
  template<typename... Args>
  void emplace_back(Args&&... args) {
    if (endptr == storageend) {
      if (capacity() != size()) {
        __builtin_unreachable();
      }
      [[unlikely]];
      expand();
    }
    new (endptr) T(std::forward<Args>(args)...);
    ++endptr;
    ++msize;
  }
  T& front() {
    return *beginptr;
  }
  T& back() {
    return endptr[-1];
  }
  void pop_back() {
    --endptr;
    --msize;
    endptr->~T();
  }
};

} // namespace moolib
