/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "synchronization.h"

#include "memfd.h"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <stdlib.h>
#include <vector>
#include <algorithm>

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
      //std::free(ptr);
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
    // Header* r = (Header*)aligned_alloc(64, size);
    // new (r) Header();
    // r->capacity = size - sizeof(Header);
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

inline std::unordered_map<size_t, size_t> allocsizes;
inline std::mutex allocmutex;
inline size_t nAllocs = 0;

template<typename Header, typename Data>
Header* allocate(size_t n) {
  // std::lock_guard l(allocmutex);
  // allocsizes[n] += 1;
  // if (++nAllocs % 100000 == 0) {
  //   printf("%d allocations (sizeof Header %d)\n", nAllocs, sizeof(Header));
  //   std::vector<std::pair<size_t, size_t>> sorted;
  //   for (auto& [k, v] : allocsizes) {
  //     sorted.emplace_back(v, k);
  //   }
  //   std::sort(sorted.begin(), sorted.end());
  //   for (auto& [v, k] : sorted) {
  //     printf(" %d  x%d\n", k, v);
  //   }
  // }
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
    // Header* h = (Header*)aligned_alloc(64, (sizeof(Header) + sizeof(Data) * n + 63) / 64 * 64);
    // new (h) Header();
    // h->capacity = n;
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
  if (n == 64) {
    allocimpl::Storage<Header, Data, 64>::get().deallocate(ptr);
  } else if (n == 256) {
    allocimpl::Storage<Header, Data, 256>::get().deallocate(ptr);
  } else if (n == 1024) {
    allocimpl::Storage<Header, Data, 1024>::get().deallocate(ptr);
  } else if (n == 4096) {
    allocimpl::Storage<Header, Data, 4096>::get().deallocate(ptr);
  } else {
    if (n <= 4096) {
      printf("n is %d\n", n);
      std::abort();
    }
    memfdAllocator.deallocate(ptr, ptr->capacity + sizeof(Header));
  }
}
template<typename Data, typename Header>
Data* dataptr(Header* ptr) {
  return (Data*)(ptr + 1);
}

} // namespace rpc
