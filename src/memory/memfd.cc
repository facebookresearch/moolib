
#include "memfd.h"
#include "synchronization.h"

#include "fmt/printf.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <shared_mutex>
#include <system_error>
#include <unordered_map>
#include <utility>

#include <sys/mman.h>
#include <unistd.h>

#undef assert
//#define assert(x) (bool(x) ? 0 : (printf("assert failure %s:%d\n", __FILE__, __LINE__), std::abort(), 0))
//#define assert(x) (bool(x) ? 0 : (__builtin_unreachable(), 0))
#define assert(x)

namespace rpc {

namespace memfd {

Memfd::~Memfd() {
  if (base != nullptr) {
    munmap(base, size);
    base = nullptr;
  }
  if (fd != -1) {
    // fmt::printf("memfd close %d\n", fd);
    close(fd);
    fd = -1;
  }
}

Memfd Memfd::create(size_t size) {
  int fd = memfd_create("Memfd", MFD_CLOEXEC);
  if (fd == -1) {
    throw std::system_error(errno, std::generic_category(), "memfd_create");
  }
  if (ftruncate(fd, size)) {
    close(fd);
    throw std::system_error(errno, std::generic_category(), "ftruncate");
  }

  // fmt::printf("memfd create %d\n", fd);

  return map(fd, size);
}

Memfd Memfd::map(int fd, size_t size) {
  void* base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (!base || base == MAP_FAILED) {
    close(fd);
    throw std::system_error(errno, std::generic_category(), "mmap");
  }

  // fmt::printf("memfd map %d\n", fd);

  Memfd r;
  r.fd = fd;
  r.size = size;
  r.base = base;
  return r;
}

struct alignas(64) AllocatorImpl {
  static_assert(sizeof(long) == sizeof(size_t));
  static constexpr size_t sizeBits = 8 * sizeof(size_t);
  size_t bucketBits = 0;
  std::array<uint8_t, sizeBits> subBucketBits;

  std::array<uintptr_t, 8 * sizeBits> spanCurrent{};
  std::array<uintptr_t*, 8 * sizeBits> spanBegin{};
  std::array<uintptr_t*, 8 * sizeBits> spanEnd{};

  ~AllocatorImpl() {
    for (auto& v : spanBegin) {
      if (v) {
        std::free(v);
      }
    }
    spanCurrent.fill(0);
    spanBegin.fill(nullptr);
    spanEnd.fill(nullptr);
  }

  void expandPushSpan(size_t index, uintptr_t value) {
    uintptr_t* begin = spanBegin[index];
    assert((spanCurrent[index] & 7) == 0);
    size_t n = (uintptr_t*)spanCurrent[index] - begin;
    size_t newSize = std::max(n + n / (size_t)2, (size_t)3) + 1;
    uintptr_t* ptr = (uintptr_t*)std::malloc(sizeof(uintptr_t) * newSize);
    uintptr_t* end = ptr + newSize;
    std::memcpy(ptr, begin, sizeof(uintptr_t) * n);
    if (n == 0) {
      value |= 1;
    }
    ptr[n] = value;
    ++n;
    std::memset(ptr + n, 0, sizeof(uintptr_t) * (newSize - n));
    assert(newSize > n);
    ptr[newSize - 1] = 2;
    spanBegin[index] = ptr;
    spanCurrent[index] = (uintptr_t)(ptr + n);
    spanEnd[index] = ptr + newSize;
    if (begin) {
      std::free(begin);
    }
  }

  void pushSpan(size_t index, uintptr_t value) {
    uintptr_t cur = spanCurrent[index];
    if (!cur || *(uintptr_t*)cur == 2) {
      [[unlikely]];
      expandPushSpan(index, value);
    } else {
      assert(cur < (uintptr_t)spanEnd[index] - 8);
      uintptr_t lowbits = cur & (size_t)3;
      value |= lowbits;
      cur &= ~(size_t)3;
      *(uintptr_t*)cur = value;
      spanCurrent[index] = cur + sizeof(uintptr_t);
    }
  }

  static size_t getSizeFor(size_t index, size_t subindex) {
    return (1ul << index) + ((1ul << (index - 3)) * (1 + subindex));
  }

  template<typename T>
  static size_t findFirstSetBit(T v) {
    static_assert(sizeof(T) == sizeof(long));
    return __builtin_ctzl(v);
  }
  template<typename T>
  static size_t findLastSetBit(T v) {
    static_assert(sizeof(T) == sizeof(long));
    return (sizeof(T) * 8 - 1) ^ __builtin_clzl(v);
  }

  template<bool isDeallocation = false>
  void addArea(void* ptr, size_t size) {
    uintptr_t address = (uintptr_t)ptr;
    while (true) {
      assert((address & (alignof(std::max_align_t) - 1)) == 0);
      assert(size >= alignof(std::max_align_t));
      assert((size & (alignof(std::max_align_t) - 1)) == 0);
      size_t index;
      size_t subindex;
      size_t nsize = size;
      if constexpr (!isDeallocation) {
        index = findLastSetBit(size);
        nsize = size & ((size_t)-1 << (index - 3));
        index = index - (nsize & (nsize - 1) ? 0 : 1);
      } else {
        index = findLastSetBit(nsize - 1);
      }
      subindex = ((nsize - 1) >> (index - 3)) & 7;
      assert(nsize == getSizeFor(index, subindex));
      assert(nsize <= size);
      bucketBits |= 1ul << index;
      subBucketBits[index] |= 1ul << subindex;
      assert(nsize == getSizeFor(index, subindex));
      pushSpan(index * 8u + subindex, address);
      address += nsize;
      size -= nsize;
      if (isDeallocation || size == 0) {
        assert(size == 0);
        break;
      }
    }
  }

  static size_t allocationSizeFor(size_t size) {
    size = (size + alignof(std::max_align_t) - 1) & ~(alignof(std::max_align_t) - 1);
    size_t index = findLastSetBit(size - 1);
    size_t subindex = ((size - 1) >> (index - 3)) & 7;
    return getSizeFor(index, subindex);
  }

  [[gnu::always_inline]] std::pair<void*, size_t> allocate(size_t size) {
    size = (size + alignof(std::max_align_t) - 1) & ~(alignof(std::max_align_t) - 1);
    size_t index = findLastSetBit(size - 1);
    size_t subindex = ((size - 1) >> (index - 3)) & 7;

    assert(getSizeFor(index, subindex) >= size);

    size_t bits = bucketBits;

    bool perfectMatch = false;
    bool partialMatch = false;
    size_t freeIndex;
    size_t freeSubindex;
    uint8_t subBits;
    size_t allocSize = getSizeFor(index, subindex);
    size_t spanSize;
    size_t fullindex;
    uintptr_t cur;
    uintptr_t ptr;

    if (bits & (1ul << index)) {
      [[likely]];
      subBits = subBucketBits[index];
      if (subBits & (1ul << subindex)) {
        [[likely]];
        freeIndex = index;
        freeSubindex = subindex;
        fullindex = freeIndex * 8u + freeSubindex;
        perfectMatch = true;
        spanSize = allocSize;
      } else {
        subBits >>= subindex;
        subBits <<= subindex;
        if (subBits != 0) {
          freeIndex = index;
          freeSubindex = findFirstSetBit((size_t)subBits);
          spanSize = getSizeFor(freeIndex, freeSubindex);
          partialMatch = true;
        }
      }
    }
    if (!perfectMatch && !partialMatch) {
      bits >>= index + 1;
      bits <<= index + 1;
      if (bits == 0) {
        [[unlikely]];
        return {nullptr, 0};
      }
      freeIndex = findFirstSetBit(bits);
      subBits = subBucketBits[freeIndex];
      freeSubindex = findFirstSetBit((size_t)subBits);
      spanSize = getSizeFor(freeIndex, freeSubindex);
      assert(spanSize > allocSize);
    }
    assert(subBits);

    fullindex = freeIndex * 8u + freeSubindex;

    assert(spanSize >= allocSize);

    cur = spanCurrent[fullindex];
    assert((cur & 7) == 0);
    cur -= sizeof(uintptr_t);
    ptr = *(uintptr_t*)cur;
    assert((ptr & 14) == 0);
    assert(ptr > 0);
    if (ptr & 1) {
      [[unlikely]];
      ptr &= ~(size_t)1;
      assert(cur == (uintptr_t)spanBegin[fullindex]);
      cur |= 1;
      subBucketBits[freeIndex] &= ~((size_t)1 << freeSubindex);
      if (subBucketBits[freeIndex] == 0) {
        bucketBits &= ~((size_t)1 << freeIndex);
      }
    }
    assert(cur >= (uintptr_t)spanBegin[fullindex]);
    spanCurrent[fullindex] = cur;
    if (!perfectMatch) {
      addArea<false>((void*)(ptr + allocSize), spanSize - allocSize);
    }
    return {(void*)ptr, allocSize};
  }

  void deallocate(void* ptr, size_t size) {
    addArea<true>(ptr, size);
  }
};

struct MemfdInfo {
  Memfd memfd;
};

struct MemfdAllocatorImpl {
  SpinMutex allocatorMutex;
  SharedSpinMutex expandMutex;
  AllocatorImpl allocator;
  std::unordered_map<int, MemfdInfo> fdToMemfd;
  std::vector<MemfdInfo*> memfds;
  size_t allocated = 0;
  MemfdInfo* getMemfdForAddressLocked(void* ptr) {
    auto i = std::lower_bound(memfds.begin(), memfds.end(), (uintptr_t)ptr, [](auto& a, uintptr_t b) {
      return (uintptr_t)a->memfd.base <= b;
    });
    if (i == memfds.begin()) {
      return nullptr;
    }
    i = std::prev(i);
    auto& v = **i;
    if ((uintptr_t)ptr >= (uintptr_t)v.memfd.base && (uintptr_t)ptr < (uintptr_t)v.memfd.base + v.memfd.size) {
      return &v;
    } else {
      return nullptr;
    }
  }
  AddressInfo getAddressInfo(void* ptr) {
    std::shared_lock l(expandMutex);
    MemfdInfo* m = getMemfdForAddressLocked(ptr);
    if (m) {
      AddressInfo r;
      r.fd = m->memfd.fd;
      r.fdSize = m->memfd.size;
      r.offset = (uintptr_t)ptr - (uintptr_t)m->memfd.base;
      return r;
    } else {
      return {};
    }
  }
  void expand(size_t size) {
    size = allocator.allocationSizeFor(size) + 128;
    size_t memsize = std::max(allocated, size);
    if (memsize > size + 1024 * 1024 * 256) {
      if (allocated / 2 > size) {
        memsize = allocated / 2;
      }
    }
    if (memsize < 1024 * 1024 * 32) {
      memsize = 1024 * 1024 * 32;
    }
    if (memsize < size + 1024 * 1024) {
      memsize += 1024 * 1024;
    }
    memsize = (memsize + 1024 * 1024 * 2 - 1) / (1024u * 1024 * 2) * (1024u * 1024 * 2);
    // fmt::printf("memfd create %d (%dM) bytes\n", memsize, memsize / 1024 / 1024);
    auto memfd = Memfd::create(memsize);
    void* base = memfd.base;
    allocator.addArea((char*)base + 64, memfd.size - 128);
    auto* ptr = &fdToMemfd[memfd.fd];
    *ptr = {};
    ptr->memfd = std::move(memfd);
    memfds.insert(
        std::lower_bound(
            memfds.begin(), memfds.end(), (uintptr_t)base,
            [](auto& a, uintptr_t b) { return (uintptr_t)a->memfd.base < b; }),
        ptr);
    allocated += memsize;
    // fmt::printf("allocated -> %dM\n", allocated / 1024 / 1024);
  }
  [[gnu::noinline]] [[gnu::cold]] std::pair<void*, size_t> expandAndAllocate(size_t size) {
    std::lock_guard l(expandMutex);
    expand(size);
    return allocator.allocate(size);
  }
  std::pair<void*, size_t> allocate(size_t size) {
    std::lock_guard l(allocatorMutex);
    auto r = allocator.allocate(size);
    if (r.first) {
      [[likely]];
      return r;
    } else {
      return expandAndAllocate(size);
    }
  }
  void deallocate(void* ptr, size_t size) {
    std::lock_guard l(allocatorMutex);
    allocator.deallocate(ptr, size);
  }

  std::pair<void*, size_t> getMemfd(int fd) {
    std::shared_lock l(expandMutex);
    auto i = fdToMemfd.find(fd);
    if (i != fdToMemfd.end()) {
      return {i->second.memfd.base, i->second.memfd.size};
    } else {
      return {nullptr, 0};
    }
  }
};

MemfdAllocator::MemfdAllocator() {
  impl = std::make_unique<MemfdAllocatorImpl>();
}
MemfdAllocator::~MemfdAllocator() {}
AddressInfo MemfdAllocator::getAddressInfo(void* ptr) {
  return impl->getAddressInfo(ptr);
}
std::pair<void*, size_t> MemfdAllocator::allocate(size_t size) {
  return impl->allocate(size);
}
void MemfdAllocator::deallocate(void* ptr, size_t size) {
  impl->deallocate(ptr, size);
}

std::pair<void*, size_t> MemfdAllocator::getMemfd(int fd) {
  return impl->getMemfd(fd);
}

} // namespace memfd

} // namespace rpc
