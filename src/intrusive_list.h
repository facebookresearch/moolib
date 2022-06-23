#pragma once

#include <cstddef>
#include <iterator>

namespace moolib {

template<typename T>
struct IntrusiveListLink {
  T* prev = nullptr;
  T* next = nullptr;
};

template<typename T, IntrusiveListLink<T> T::*link>
struct IntrusiveList {
private:
  T head;
  static T*& next(T* at) noexcept {
    return (at->*link).next;
  }
  static T*& prev(T* at) noexcept {
    return (at->*link).prev;
  }

public:
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;

  struct iterator {
  private:
    T* ptr = nullptr;

  public:
    iterator() = default;
    iterator(T* ptr) : ptr(ptr) {}

    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using iterator_category = std::bidirectional_iterator_tag;

    T& operator*() const noexcept {
      return *ptr;
    }
    T* operator->() const noexcept {
      return ptr;
    }
    iterator& operator++() noexcept {
      ptr = next(ptr);
      return *this;
    }
    iterator operator++(int) noexcept {
      iterator r = (*this);
      ptr = next(ptr);
      return r;
    }
    iterator& operator--() noexcept {
      ptr = prev(ptr);
      return *this;
    }
    iterator operator--(int) noexcept {
      iterator r = (*this);
      ptr = prev(ptr);
      return r;
    }
    bool operator==(iterator n) const noexcept {
      return ptr == n.ptr;
    }
    bool operator!=(iterator n) const noexcept {
      return ptr != n.ptr;
    }
  };

  IntrusiveList() noexcept {
    prev(&head) = &head;
    next(&head) = &head;
  }

  iterator begin() noexcept {
    return iterator(next(&head));
  }
  iterator end() noexcept {
    return iterator(&head);
  }
  size_t size() = delete;
  constexpr size_t max_size() = delete;
  bool empty() const noexcept {
    return next((T*)&head) == &head;
  }

  void clear() noexcept {
    prev(&head) = &head;
    next(&head) = &head;
  }
  iterator insert(iterator at, T& item) noexcept {
    T* nextItem = &*at;
    T* prevItem = prev(&*at);
    prev(nextItem) = &item;
    next(prevItem) = &item;
    next(&item) = nextItem;
    prev(&item) = prevItem;
    return at;
  }
  static iterator erase(iterator at) noexcept {
    T* nextItem = next(&*at);
    T* prevItem = prev(&*at);
    prev(nextItem) = prevItem;
    next(prevItem) = nextItem;
    prev(&*at) = nullptr;
    next(&*at) = nullptr;
    return at;
  }
  static void erase(T& item) noexcept {
    erase(iterator(&item));
  }
  iterator push_front(T& item) noexcept {
    return insert(begin(), item);
  }
  iterator push_back(T& item) noexcept {
    return insert(end(), item);
  }
  void pop_front() noexcept {
    erase(begin());
  }
  void pop_back() noexcept {
    erase(prev(&head));
  }
  T& front() noexcept {
    return *next(&head);
  }
  T& back() noexcept {
    return *prev(&head);
  }
};

} // namespace moolib
