/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "synchronization.h"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <stdlib.h>

namespace rpc {

namespace allocimpl {

template<typename Header, typename Data, size_t Size>
struct Storage {
  static constexpr size_t size = (Size + 63) / 64 * 64;
  Header* freelist = nullptr;
  size_t freelistSize = 0;

  ~Storage() {
    for (Header* ptr = freelist; ptr;) {
      Header* next = ptr->next;
      std::free(ptr);
      ptr = next;
    }
  }

  Header* allocate() {
    static_assert(alignof(Header) <= 64 && alignof(Data) <= 64 && alignof(Data) <= sizeof(Header));
    Header* r = freelist;
    if (!r) {
      r = (Header*)aligned_alloc(64, size);
      new (r) Header();
      r->capacity = size - sizeof(Header);
    } else {
      --freelistSize;
      freelist = r->next;
      if (r->capacity != size - sizeof(Header)) {
        std::abort();
      }
      if (r->refcount != 0) {
        std::abort();
      }
    }
    if (r->refcount != 0) {
      std::abort();
    }
    return r;
  }
  void deallocate(Header* ptr) {
    if (ptr->refcount != 0) {
      std::abort();
    }
    if (freelistSize >= 1024 * 1024 / Size) {
      std::free(ptr);
      return;
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
    Header* h = (Header*)aligned_alloc(64, (sizeof(Header) + sizeof(Data) * n + 63) / 64 * 64);
    new (h) Header();
    h->capacity = n;
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
    if (n <= 4096 || ptr->refcount != 0) {
      std::abort();
    }
    std::free(ptr);
  }
}
template<typename Data, typename Header>
Data* dataptr(Header* ptr) {
  return (Data*)(ptr + 1);
}

} // namespace rpc
