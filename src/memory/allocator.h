/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "synchronization.h"

#include "memfd.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <stdlib.h>
#include <vector>

namespace rpc {

inline memfd::MemfdAllocator memfdAllocator;

namespace allocimpl {

template<typename Header, typename Data, size_t Size>
struct Storage {
  static constexpr size_t size = (Size + 63) / 64 * 64;
  Header* freelist = nullptr;
  size_t freelistSize = 0;

  struct alignas(64) GlobalList {
    SpinMutex mutex;
    std::vector<std::pair<Header*, size_t>> list;
  };

  ~Storage() {
    for (Header* ptr = freelist; ptr;) {
      Header* next = ptr->next;
      memfdAllocator.deallocate(ptr, ptr->capacity + sizeof(Header));
      ptr = next;
    }
  }

  inline static GlobalList global;

  Header* allocateFromGlobal() {
    std::unique_lock l(global.mutex);
    if (!global.list.empty()) {
      freelist = global.list.back().first;
      freelistSize = global.list.back().second;
      global.list.pop_back();
      l.unlock();
      return allocate();
    }
    l.unlock();
    auto a = memfdAllocator.allocate(size);
    Header* r = (Header*)a.first;
    new (r) Header();
    r->capacity = size - sizeof(Header);
    return r;
  }

  Header* allocate() {
    static_assert(alignof(Header) <= 64 && alignof(Data) <= 64 && alignof(Data) <= sizeof(Header));
    Header* r = freelist;
    if (r) {
      [[likely]];
      --freelistSize;
      freelist = r->next;
      return r;
    } else {
      return allocateFromGlobal();
    }
  }
  void moveFreelistToGlobal() {
    std::unique_lock l(global.mutex);
    global.list.push_back({freelist, freelistSize});
    l.unlock();
    freelist = nullptr;
    freelistSize = 0;
  }
  void deallocate(Header* ptr) {
    if (freelistSize >= std::min<size_t>(1024 * 1024 / Size, 128)) {
      [[unlikely]];
      moveFreelistToGlobal();
    }
    ++freelistSize;
    ptr->next = freelist;
    freelist = ptr;
  }

  static Storage& get() {
    thread_local Storage storage;
    return storage;
  }
};

} // namespace allocimpl

template<typename Header, typename Data>
Header* allocate(size_t n) {
  constexpr size_t overhead = sizeof(Header);
  if (n + overhead <= 64) {
    return allocimpl::Storage<Header, Data, 64>::get().allocate();
  } else if (n + overhead <= 256) {
    return allocimpl::Storage<Header, Data, 256>::get().allocate();
  } else if (n + overhead <= 1024) {
    return allocimpl::Storage<Header, Data, 1024>::get().allocate();
  } else if (n + overhead <= 4096) {
    return allocimpl::Storage<Header, Data, 4096>::get().allocate();
  } else {
    auto a = memfdAllocator.allocate((sizeof(Header) + sizeof(Data) * n + 63) / 64 * 64);
    Header* h = (Header*)a.first;
    new (h) Header();
    h->capacity = a.second - sizeof(Header);
    return h;
  }
}
template<typename Header, typename Data>
void deallocate(Header* ptr) {
  const size_t n = ptr->capacity + sizeof(Header);
  switch (n) {
  case 64:
    allocimpl::Storage<Header, Data, 64>::get().deallocate(ptr);
    break;
  case 256:
    allocimpl::Storage<Header, Data, 256>::get().deallocate(ptr);
    break;
  case 1024:
    allocimpl::Storage<Header, Data, 1024>::get().deallocate(ptr);
    break;
  case 4096:
    allocimpl::Storage<Header, Data, 4096>::get().deallocate(ptr);
    break;
  default:
    memfdAllocator.deallocate(ptr, ptr->capacity + sizeof(Header));
  }
}
template<typename Data, typename Header>
Data* dataptr(Header* ptr) {
  return (Data*)(ptr + 1);
}

} // namespace rpc
