/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "allocator.h"
#include "tensor.h"

#include <cstddef>
#include <cstdlib>
#include <new>

namespace rpc {

struct TensorRef {
  Tensor tensor;
};

struct Buffer {
  Buffer* next{nullptr};
  size_t capacity = 0;
  size_t size = 0;
  std::atomic_int refcount = 0;
  uint32_t nTensors = 0;
  std::byte* data() {
    return dataptr<std::byte>(this);
  }
  template<typename T, typename RT = T*, typename P>
  static RT roundUpFor(P ptr) {
    uintptr_t v = (uintptr_t)(std::byte*)ptr;
    constexpr auto alignment = alignof(T);
    static_assert(alignment <= 64);
    if (alignment <= alignof(std::remove_pointer_t<P>)) {
      return (RT)v;
    }
    return (RT)((v + alignment - 1) / alignment * alignment);
  }
  size_t* tensorMetaDataOffsets() {
    return roundUpFor<size_t>(data() + size);
  }
  TensorRef* tensors() {
    return roundUpFor<TensorRef>(tensorMetaDataOffsets() + nTensors);
  }
  static size_t getAllocSize(size_t size, size_t nTensors) {
    uintptr_t ptr = 0;
    uintptr_t offsets = roundUpFor<size_t, uintptr_t>(ptr + size);
    uintptr_t tensors = roundUpFor<TensorRef, uintptr_t>(offsets + sizeof(size_t) * nTensors);
    return tensors + sizeof(TensorRef) * nTensors - ptr;
  }
};

inline void destroyBuffer(Buffer* buffer) noexcept {
  if (buffer->nTensors) {
    auto* tensors = buffer->tensors();
    for (size_t i = buffer->nTensors; i;) {
      --i;
      tensors[i].~TensorRef();
    }
    auto* offsets = buffer->tensorMetaDataOffsets();
    for (size_t i = buffer->nTensors; i;) {
      --i;
      offsets[i].~size_t();
    }
    buffer->nTensors = 0;
  }
}

inline void shrinkBuffer(Buffer* buffer, size_t size, size_t nTensors) {
  auto* tensors = buffer->tensors();
  for (size_t i = buffer->nTensors; i != nTensors;) {
    --i;
    tensors[i].~TensorRef();
  }
  auto* offsets = buffer->tensorMetaDataOffsets();
  for (size_t i = buffer->nTensors; i != nTensors;) {
    --i;
    offsets[i].~size_t();
  }
  buffer->nTensors = nTensors;
  buffer->size = size;
}

struct BufferHandle {
  Buffer* buffer_ = nullptr;
  BufferHandle() = default;
  BufferHandle(std::nullptr_t) noexcept {}
  explicit BufferHandle(Buffer* buffer) noexcept : buffer_(buffer) {}
  BufferHandle(const BufferHandle&) = delete;
  BufferHandle& operator=(const BufferHandle&) = delete;
  BufferHandle(BufferHandle&& n) noexcept {
    buffer_ = n.buffer_;
    n.buffer_ = nullptr;
  }
  BufferHandle& operator=(BufferHandle&& n) noexcept {
    std::swap(buffer_, n.buffer_);
    return *this;
  }
  ~BufferHandle() {
    if (buffer_) {
      destroyBuffer(buffer_);
      deallocate<Buffer, std::byte>(buffer_);
    }
  }
  explicit operator bool() const noexcept {
    return buffer_;
  }
  Buffer* operator->() const noexcept {
    return buffer_;
  }
  operator Buffer*() const noexcept {
    return buffer_;
  }
  Buffer* release() noexcept {
    Buffer* r = buffer_;
    buffer_ = nullptr;
    return r;
  }
};
struct SharedBufferHandle {
  Buffer* buffer_ = nullptr;
  SharedBufferHandle() = default;
  SharedBufferHandle(std::nullptr_t) noexcept {}
  explicit SharedBufferHandle(Buffer* buffer) noexcept : buffer_(buffer) {
    if (buffer_) {
      if (buffer->refcount != 0) {
        std::abort();
      }
      addref();
    }
  }
  SharedBufferHandle(const SharedBufferHandle& n) noexcept {
    buffer_ = n.buffer_;
    if (buffer_) {
      addref();
    }
  }
  SharedBufferHandle& operator=(const SharedBufferHandle& n) noexcept {
    buffer_ = n.buffer_;
    if (buffer_) {
      addref();
    }
    return *this;
  }
  SharedBufferHandle(SharedBufferHandle&& n) noexcept {
    buffer_ = n.buffer_;
    n.buffer_ = nullptr;
  }
  SharedBufferHandle& operator=(SharedBufferHandle&& n) noexcept {
    std::swap(buffer_, n.buffer_);
    return *this;
  }
  ~SharedBufferHandle() {
    if (buffer_ && decref() == 0) {
      destroyBuffer(buffer_);
      deallocate<Buffer, std::byte>(buffer_);
    }
  }
  explicit operator bool() const noexcept {
    return buffer_;
  }
  Buffer* operator->() const noexcept {
    return buffer_;
  }
  operator Buffer*() const noexcept {
    return buffer_;
  }
  int addref() noexcept {
    return buffer_->refcount.fetch_add(1, std::memory_order_acquire) + 1;
  }
  int decref() noexcept {
    return buffer_->refcount.fetch_sub(1) - 1;
  }
  Buffer* release() noexcept {
    Buffer* r = buffer_;
    buffer_ = nullptr;
    return r;
  }
  void acquire(Buffer* buffer) noexcept {
    buffer_ = buffer;
  }
};

inline BufferHandle makeBuffer(size_t size, size_t nTensors) noexcept {
  size_t allocsize = size;
  if (nTensors) {
    allocsize = Buffer::getAllocSize(size, nTensors);
  }
  BufferHandle buffer{allocate<Buffer, std::byte>(allocsize)};
  buffer->size = size;
  buffer->nTensors = nTensors;
  if (nTensors) {
    auto* offsets = buffer->tensorMetaDataOffsets();
    for (size_t i = 0; i != nTensors; ++i) {
      new (offsets + i) size_t{};
    }
    auto* tensors = buffer->tensors();
    for (size_t i = 0; i != nTensors; ++i) {
      new (tensors + i) TensorRef{};
    }
  }
  return buffer;
}

} // namespace rpc
