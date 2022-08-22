#pragma once

#include <cstdlib>
#include <utility>
#include <memory>
#include <atomic>
#include <array>

namespace rpc {
namespace memfd {

struct Memfd {
  int fd = -1;
  void* base = nullptr;
  size_t size = 0;

  Memfd() = default;
  ~Memfd();
  Memfd(const Memfd&) = delete;
  Memfd(Memfd&& n) {
    std::swap(fd, n.fd);
    std::swap(base, n.base);
    std::swap(size, n.size);
  }
  Memfd& operator=(const Memfd&) = delete;
  Memfd& operator=(Memfd&& n) {
    std::swap(fd, n.fd);
    std::swap(base, n.base);
    std::swap(size, n.size);
    return *this;
  }

  static Memfd create(size_t size);
  static Memfd map(int fd, size_t size);
};

struct AddressInfo {
  int fd = -1;
  size_t fdSize = 0;
  size_t offset = 0;
};

struct MemfdAllocatorImpl;
struct MemfdAllocator {
  std::unique_ptr<MemfdAllocatorImpl> impl;
  MemfdAllocator();
  ~MemfdAllocator();
  std::pair<void*, size_t> getMemfd(int fd);
  AddressInfo getAddressInfo(void* ptr);
  std::pair<void*, size_t> allocate(size_t size);
  void deallocate(void* ptr, size_t size);
  void debugInfo();
};

}

}