
#include "memfd.h"
#include "synchronization.h"
#include "vector.h"

#include "fmt/printf.h"

#include <vector>
#include <algorithm>
#include <utility>
#include <system_error>
#include <memory>
#include <cstddef>

#include <unistd.h>
#include <sys/mman.h>

#undef assert
#define assert(x) (bool(x) ? 0 : (printf("assert failure %s:%d\n", __FILE__, __LINE__), std::abort(), 0))
//#define assert(x)

namespace rpc {

namespace memfd {

Memfd::~Memfd() {
  if (base != nullptr) {
    munmap(base, size);
    base = nullptr;
  }
  if (fd != -1) {
    fmt::printf("memfd close %d\n", fd);
    close(fd);
    fd = -1;
  }
}

Memfd Memfd::create(size_t size) {
  int fd = memfd_create("Memfd", MFD_CLOEXEC);
  if (fd == -1) {
    throw std::system_error(errno, std::generic_category());
  }
  if (ftruncate(fd, size)) {
    close(fd);
    throw std::system_error(errno, std::generic_category());
  }

  void* base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (!base || base == MAP_FAILED) {
    close(fd);
    throw std::system_error(errno, std::generic_category());
  }

  fmt::printf("memfd create %d\n", fd);

  Memfd r;
  r.fd = fd;
  r.size = size;
  r.base = base;
  return r;
}

Memfd Memfd::map(int fd, size_t size) {
  void* base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (!base || base == MAP_FAILED) {
    close(fd);
    throw std::system_error(errno, std::generic_category());
  }

  fmt::printf("memfd map %d\n", fd);

  Memfd r;
  r.fd = fd;
  r.size = size;
  r.base = base;
  return r;
}

struct AllocatorImpl {

  std::unordered_map<uintptr_t, uintptr_t> beginMap;
  std::unordered_map<uintptr_t, uintptr_t> endMap;
  
  static_assert(sizeof(long) == sizeof(size_t));
  static constexpr size_t sizeBits = 8 * sizeof(size_t);
  size_t bucketBits = 0;
  std::array<uint8_t, sizeBits> subBucketBits;

  std::array<std::vector<uintptr_t>, 8 * sizeBits> spans;

  size_t totalAvailableSize = 0;
  size_t totalAllocatedSize = 0;

  size_t getSizeFor(size_t index, size_t subindex) {
    return (1ul << index) + ((1ul << (index - 3)) * (1 + subindex));
  }

  template<typename T>
  size_t ctz(T v) {
    static_assert(sizeof(T) == sizeof(long));
    return __builtin_ctzl(v);
  }
  template<typename T>
  size_t clz(T v) {
    static_assert(sizeof(T) == sizeof(long));
    return __builtin_clzl(v);
  }

  template<bool isDeallocation = false>
  void addArea(void* ptr, size_t size) {
    while (size >= (isDeallocation ? 0 : alignof(std::max_align_t))) {
      assert(((uintptr_t)ptr & (alignof(std::max_align_t) - 1)) == 0);
      assert(size >= alignof(std::max_align_t));
      size_t index;
      size_t subindex;
      size_t nsize;
      if (!isDeallocation) {
        index = sizeBits - 1 - clz(size);
        subindex = (size >> (index - 3)) & 7;
        nsize = (1ul << index) | ((1ul << (index - 3)) * subindex);
        index = sizeBits - 1 - clz(nsize - 1);
        subindex = ((nsize - 1) >> (index - 3)) & 7;
        //fmt::printf("adding area of size %d (lost %d) at %#x  (index %d subindex %d)\n", nsize, size - nsize, (uintptr_t)ptr, index, subindex);
      } else {
        index = sizeBits - 1 - clz((size - 1) | 8);
        subindex = ((size - 1) >> (index - 3)) & 7;
        assert(size == getSizeFor(index, subindex));
        nsize = size;
        //fmt::printf("deallocate area of size %d (lost %d) at %#x  (index %d subindex %d)\n", nsize, size - nsize, (uintptr_t)ptr, index, subindex);
      }
      //fmt::printf("index %d subindex %d\n", index, subindex);
      bucketBits |= 1ul << index;
      subBucketBits[index] |= 1ul << subindex;
      //fmt::printf("add address %#x to full index %d\n", (uintptr_t)ptr, index * 8u + subindex);
      spans[index * 8u + subindex].push_back((uintptr_t)ptr);
      ptr = (void*)((char*)ptr + nsize);
      size -= nsize;
      if (isDeallocation) {
        assert(size == 0);
        break;
      }
    }
  }

  std::pair<void*, size_t> allocate(size_t size) {
    size = (size + alignof(std::max_align_t) - 1) & ~(alignof(std::max_align_t) - 1);
    size_t index = sizeBits - 1 - clz((size - 1) | 8);
    size_t subindex = ((size - 1) >> (index - 3)) & 7;

    //fmt::printf("allocate size %d, index %d, subindex %d\n", size, index, subindex);

    assert(getSizeFor(index, subindex) >= size);

    size_t bits = bucketBits;

    // if (bits & (1ul << index)) {
    //   uint8_t subBits = subBucketBits[index];
    //   if (subBits & (1ul << subindex)) {
    //       fmt::printf("perfect match full index %d\n", index * 8u + subindex);
    //       auto& vec = spans[index * 8u + subindex];
    //       assert(!vec.empty());
    //       auto r = vec.back();
    //       fmt::printf("%d %d  %#x %#x\n", getSizeFor(index, subindex), r.end - r.begin, r.begin, r.end);
    //       assert(r.begin + getSizeFor(index, subindex) == r.end);
    //       vec.pop_back();
    //       fmt::printf("found perfect match for allocation of size %d\n", size);
    //       if (vec.empty()) {
    //         fmt::printf("subindex %d is now empty\n", subindex);
    //         subBits &= ~(1ul << subindex);
    //         subBucketBits[subindex] = subBits;
    //         if (subBits == 0) {
    //           bucketBits &= ~(1ul << index);
    //           fmt::printf("index %d is now empty\n", index);
    //         }
    //       }
    //       return {(void*)r.begin, r.end - r.begin};
    //   }
    // }

    bool perfectMatch = false;
    bool partialMatch = false;
    size_t freeIndex;
    size_t freeSubindex;
    uint8_t subBits;
    size_t allocSize = getSizeFor(index, subindex);
    size_t spanSize;

    if (bits & (1ul << index)) {
      subBits = subBucketBits[index];
      if (subBits & (1ul << subindex)) {
        freeIndex = index;
        freeSubindex = subindex;
        perfectMatch = true;
        spanSize = allocSize;
      } else {
        subBits >>= subindex;
        subBits <<= subindex;
        if (subBits != 0) {
          freeIndex = index;
          freeSubindex = ctz((size_t)subBits);
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
        //fmt::printf("no free bits for allocation of size %d\n", size);
        return {nullptr, 0};
      }
      freeIndex = ctz(bits);
      subBits = subBucketBits[freeIndex];
      freeSubindex = ctz((size_t)subBits);
      spanSize = getSizeFor(freeIndex, freeSubindex);
      assert(spanSize > allocSize);
    }
    assert(subBits);


    //fmt::printf("found free index %d %d for allocation of size %d  (alloc size %d span size %d)\n", freeIndex, freeSubindex, size, allocSize, spanSize);

    assert(spanSize >= allocSize);

    auto& vec = spans[freeIndex * 8u + freeSubindex];
    assert(!vec.empty());
    uintptr_t ptr = vec.back();
    //assert(s.begin + spanSize == s.end);
    vec.pop_back();
    if (vec.empty()) {
      //fmt::printf("subindex %d is now empty\n", freeSubindex);
      subBits &= ~(1ul << freeSubindex);
      subBucketBits[freeIndex] = subBits;
      if (subBits == 0) {
        bucketBits &= ~(1ul << freeIndex);
        //fmt::printf("index %d is now empty\n", freeIndex);
      }
    }
    if (!perfectMatch) {
      addArea<false>((void*)(ptr + allocSize), spanSize - allocSize);
    }
    totalAllocatedSize += allocSize;
    //fmt::printf("returning %#x %d\n", s.begin, allocSize);
    return {(void*)ptr, allocSize};
  }

// find highest set bit.
// 56 bits? 56 top level buckets.
// look at next 4 bits to get sub-bucket (8 sub-buckets)
// return top element
// empty ? try again at next higher bucket
//        use bitsets to speed up finding a non-empty bucket

  void deallocate(void* ptr, size_t size) {
    //fmt::printf("deallocate %p, %d\n", ptr, size);
    totalAllocatedSize -= size;
    addArea<true>(ptr, size);
  }

  void debugInfo() {
    fmt::printf("total allocated: %dM / %dM\n", totalAllocatedSize / 1024 / 1024, totalAvailableSize / 1024 / 1024);

    fmt::printf("bucketBits: %#x\n", bucketBits);
    
    size_t nChunks = 0;
    size_t sum = 0;
    for (size_t i = 0; i != spans.size(); ++i) {
      size_t index = i / 8;
      size_t subindex = i % 8;
      if (!spans[i].empty()) {
        nChunks += spans[i].size();
        sum += getSizeFor(index, subindex) * spans[i].size();
        //fmt::printf("index %d %d (size %d) has %d chunks  (sum %d)\n", index, subindex, getSizeFor(index, subindex), spans[i].size(), sum);
      }
    }
    fmt::printf(" total %d chunks, sum %dM\n", nChunks, sum / 1024 / 1024);
  }
};


struct MemfdInfo {
  Memfd memfd;
};

struct MemfdAllocatorImpl {
  std::mutex mutex;
  AllocatorImpl allocator;
  std::unordered_map<int, MemfdInfo> fdToMemfd;
  std::vector<MemfdInfo*> memfds;
  size_t allocated = 0;
  MemfdInfo* getMemfdForAddressLocked(void* ptr) {
    auto i = std::lower_bound(memfds.begin(), memfds.end(), (uintptr_t)ptr, [](auto& a , uintptr_t b) {
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
    std::lock_guard l(mutex);
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
    allocator.debugInfo();

    size_t memsize = std::max(allocated, size);
    if (memsize > size + 1024 * 1024 * 256) {
      if (allocated / 2 > size) {
        memsize = allocated / 2;
      }
    }
    if (memsize < 1024 * 1024 * 32) {
      memsize = 1024 * 1024 * 32;
    }
    memsize = (memsize + 1024 * 1024 * 2 - 1) / (1024u * 1024 * 2) * (1024u * 1024 * 2);
    if (memsize < size + 1024 * 1024) {
      memsize += 1024 * 1024;
    }
    memsize += 4096;
    fmt::printf("memfd create %d (%dM) bytes\n", memsize, memsize / 1024 / 1024);
    auto memfd = Memfd::create(memsize);
    void* base = memfd.base;
    allocator.addArea((char*)base + 64, memfd.size - 128);
    allocator.totalAvailableSize += memfd.size - 128;
    auto* ptr = &fdToMemfd[memfd.fd];
    *ptr = {};
    ptr->memfd = std::move(memfd);
    memfds.insert(std::lower_bound(memfds.begin(), memfds.end(), (uintptr_t)base, [](auto& a, uintptr_t b) {
      return (uintptr_t)a->memfd.base < b;
    }), ptr);
    allocated += memsize;
    fmt::printf("allocated -> %dM\n", allocated / 1024 / 1024);
    //if (allocated >= 1024ull * 1024 * 1024 * 4) std::abort();
  }
  [[gnu::noinline]] [[gnu::cold]]
  std::pair<void*, size_t> expandAndAllocate(size_t size) {
    std::lock_guard l(mutex);
    expand(size);
    return allocator.allocate(size);
  }
  std::pair<void*, size_t> allocate(size_t size) {
    auto r = allocator.allocate(size);
    if (r.first) {
      [[likely]];
      return r;
    } else {
      return expandAndAllocate(size);
    }
  }
  void deallocate(void* ptr, size_t size) {
    //fmt::printf("deallocate [%#x, %#x)\n", (uintptr_t)ptr, (uintptr_t)ptr + size);
    //std::lock_guard l(mutex);
    allocator.deallocate(ptr, size);
  }

  std::pair<void*, size_t> getMemfd(int fd) {
    std::lock_guard l(mutex);
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

void MemfdAllocator::debugInfo() {
  return impl->allocator.debugInfo();
}

}

}
