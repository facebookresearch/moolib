
#include "memfd.h"
#include "synchronization.h"

#include "fmt/printf.h"

#include <algorithm>
#include <utility>
#include <system_error>
#include <memory>
#include <cstddef>
#include <cstdlib>

#include <unistd.h>
#include <sys/mman.h>

#undef assert
#define assert(x) (bool(x) ? 0 : (printf("assert failure %s:%d\n", __FILE__, __LINE__), std::abort(), 0))
//#define assert(x) (bool(x) ? 0 : (__builtin_unreachable(), 0))
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
    throw std::system_error(errno, std::generic_category(), "memfd_create");
  }
  if (ftruncate(fd, size)) {
    close(fd);
    throw std::system_error(errno, std::generic_category(), "ftruncate");
  }

  fmt::printf("memfd create %d\n", fd);

  return map(fd, size);
}

Memfd Memfd::map(int fd, size_t size) {
  void* base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (!base || base == MAP_FAILED) {
    close(fd);
    throw std::system_error(errno, std::generic_category(), "mmap");
  }

  fmt::printf("memfd map %d\n", fd);

  Memfd r;
  r.fd = fd;
  r.size = size;
  r.base = base;
  return r;
}

struct alignas(64) AllocatorImpl {
  static_assert(sizeof(long) == sizeof(size_t));
  static constexpr size_t sizeBits = 8 * sizeof(size_t);
  std::atomic_size_t bucketBits = 0;
  std::array<std::atomic_uint8_t, sizeBits> subBucketBits;

  std::array<std::atomic_uintptr_t, 8 * sizeBits> spanCurrent{};
  //std::array<std::atomic_uintptr_t, 8 * sizeBits> spanTop{};
  std::array<uintptr_t*, 8 * sizeBits> spanBegin{};
  std::array<uintptr_t*, 8 * sizeBits> spanEnd{};

  std::array<SpinMutex, 8 * sizeBits> spanMutex;
  
  static_assert(sizeof(std::atomic_uintptr_t) == sizeof(uintptr_t) && alignof(std::atomic_uintptr_t) == alignof(uintptr_t));

  ~AllocatorImpl() {
    for (auto& v : spanBegin) {
      if (v) {
        std::free(v);
      }
    }
    for (auto& v : spanCurrent) {
      v = 0;
    }
    spanBegin.fill(nullptr);
    spanEnd.fill(nullptr);
  }

  void expandPushSpan(size_t cur, size_t index, uintptr_t value) {
    size_t prev = cur;
    uintptr_t* begin = spanBegin[index];
    assert((cur & 7) == 0);
    size_t n = (uintptr_t*)cur - begin;
    size_t newSize = std::max(n + n / (size_t)2, (size_t)3) + 1;
    uintptr_t* ptr = (uintptr_t*)std::malloc(sizeof(uintptr_t) * newSize);
    uintptr_t* end = ptr + newSize;
    std::memcpy(ptr, begin, sizeof(uintptr_t) * n);
    //fmt::printf("expandPushSpan, begin %#x, %d at %#x\n", (uintptr_t)begin, n, (uintptr_t)ptr);
    if (n == 0) {
      value |= 1;
    }
    ptr[n] = value;
    ++n;
    std::memset(ptr + n, 0, sizeof(uintptr_t) * (newSize - n));
    assert(newSize > n);
    ptr[newSize - 1] = 2;
    spanBegin[index] = ptr;
    cur = (uintptr_t)(ptr + n);
    spanEnd[index] = ptr + newSize;
    uintptr_t prevprev = prev;
    while (!spanCurrent[index].compare_exchange_weak(prev, cur)) {
      --n;
      intptr_t diff = prev - prevprev;
      assert(diff <= 0 && -diff <= n);
      n += diff;
      if (n == 0) {
        value |= 1;
      } else {
        value &= ~(size_t)1;
      }
      ptr[n] = value;
      ++n;
      cur = (uintptr_t)(ptr + n);
      prevprev = prev;
    }
    if (begin) {
      std::free(begin);
    }
  }

  void pushSpan(size_t index, uintptr_t value) {
    std::lock_guard l(spanMutex[index]);
    uintptr_t cur = spanCurrent[index].load(std::memory_order_relaxed);
    uintptr_t prev = cur;
retry:
    if (!cur || (*(std::atomic_uintptr_t*)cur).load(std::memory_order_relaxed) == 2) {
      [[unlikely]];
      expandPushSpan(cur, index, value);
    } else {
      assert(cur < (uintptr_t)spanEnd[index] - 8);
      uintptr_t lowbits = cur & (size_t)3;
      value |= lowbits;
      cur &= ~(size_t)3;
      (*(std::atomic_uintptr_t*)cur).store(value, std::memory_order_relaxed);
      if (!spanCurrent[index].compare_exchange_strong(prev, cur + sizeof(uintptr_t))) {
        cur = prev;
        goto retry;
      }
    }

    // fmt::printf("span %d post push:\n", index);
    // for (auto* i = spanBegin[index]; i != spanEnd[index]; ++i) {
    //   fmt::printf(" -- %#x\n", *i);
    // }
  }

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
    uintptr_t address = (uintptr_t)ptr;
    while (size >= (isDeallocation ? 0 : alignof(std::max_align_t))) {
      //fmt::printf("ptr is %#x, size %d\n", (uintptr_t)ptr, size);
      assert((address & (alignof(std::max_align_t) - 1)) == 0);
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
        //fmt::printf("adding area of size %d (lost %d) at %#x  (index %d subindex %d)\n", nsize, size - nsize, address, index, subindex);
      } else {
        index = sizeBits - 1 - clz((size - 1) | 8);
        subindex = ((size - 1) >> (index - 3)) & 7;
        assert(size == getSizeFor(index, subindex));
        nsize = size;
        //fmt::printf("deallocate area of size %d (lost %d) at %#x  (index %d subindex %d)\n", nsize, size - nsize, address, index, subindex);
      }
      //fmt::printf("index %d subindex %d\n", index, subindex);
      //fmt::printf("add address %#x to full index %d\n", address, index * 8u + subindex);
      assert(nsize == getSizeFor(index, subindex));
      pushSpan(index * 8u + subindex, address);
      bucketBits.fetch_or(1ul << index, std::memory_order_relaxed);
      subBucketBits[index].fetch_or(1ul << subindex, std::memory_order_relaxed);
      address += nsize;
      size -= nsize;
      if (isDeallocation) {
        assert(size == 0);
        break;
      }
    }
  }

  [[gnu::always_inline]]
  std::pair<void*, size_t> allocate(size_t size) {
    size = (size + alignof(std::max_align_t) - 1) & ~(alignof(std::max_align_t) - 1);
    const size_t index = sizeBits - 1 - clz((size - 1) | 8);
    const size_t subindex = ((size - 1) >> (index - 3)) & 7;

    const size_t allocSize = getSizeFor(index, subindex);

    //fmt::printf("allocate size %d, index %d, subindex %d\n", size, index, subindex);

    assert(allocSize >= size);

retryall:

    size_t bits = bucketBits;
    bool perfectMatch = false;
    bool partialMatch = false;
    size_t freeIndex;
    size_t freeSubindex;
    uint8_t subBits;
    size_t spanSize;

    if (bits & (1ul << index)) {
      [[likely]];
      subBits = subBucketBits[index];
      if (subBits & (1ul << subindex)) {
        [[likely]];
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
    if (!subBits) {
      goto retryall;
    }
    assert(subBits);

    size_t fullindex = freeIndex * 8u + freeSubindex;

    //fmt::printf("found free index %d %d for allocation of size %d  (alloc size %d span size %d)\n", freeIndex, freeSubindex, size, allocSize, spanSize);

    assert(spanSize >= allocSize);

    //fmt::printf("fullindex is %d\n", fullindex);
    uintptr_t cur = spanCurrent[fullindex].load(std::memory_order_relaxed);
    uintptr_t prev = cur;
retrythis:
    assert(cur != 0);
    if ((cur & 7) != 0) {
      goto retryall;
    }
    assert((cur & 7) == 0);
    //assert(cur > (uintptr_t)spanBegin[fullindex] && cur < (uintptr_t)spanEnd[fullindex] - 8);
    cur -= sizeof(uintptr_t);
    uintptr_t ptr = (*(std::atomic_uintptr_t*)cur).load(std::memory_order_relaxed);
    //fmt::printf("cur %#x, begin %#x, end %#x, ptr %#x\n", cur, (uintptr_t)spanBegin[fullindex], (uintptr_t)spanEnd[fullindex], ptr);
    assert((ptr & 14) == 0);
    assert(ptr > 0);
    if (ptr & 1) {
      [[unlikely]];
      ptr &= ~(size_t)1;
      //assert(cur == (uintptr_t)spanBegin[fullindex]);
      cur |= 1;
      //fmt::printf("subindex %d is now empty\n", freeSubindex);
      subBucketBits[freeIndex].fetch_and(~((size_t)1 << freeSubindex), std::memory_order_relaxed);
      if (subBucketBits[freeIndex].load(std::memory_order_relaxed) == 0) {
        bucketBits.fetch_and(~((size_t)1 << freeIndex), std::memory_order_relaxed);
        //fmt::printf("index %d is now empty\n", freeIndex);
      }
    }
    assert(cur >= (uintptr_t)spanBegin[fullindex]);
    assert(cur != 0);
    // this doesnt work unless i add a generation counter
    // that might be too much overhead, and reverting this and taking mutex might be better
    if (!spanCurrent[fullindex].compare_exchange_strong(prev, cur, std::memory_order_relaxed)) {
      cur = prev;
      goto retrythis;
    }
    if (!perfectMatch) {
      addArea<false>((void*)(ptr + allocSize), spanSize - allocSize);
    }
    totalAllocatedSize += allocSize;
    //fmt::printf("returning %#x %d\n", ptr, allocSize);
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
    for (size_t i = 0; i != spanCurrent.size(); ++i) {
      size_t index = i / 8;
      size_t subindex = i % 8;
      // if (!spans[i].empty()) {
      //   nChunks += spans[i].size();
      //   sum += getSizeFor(index, subindex) * spans[i].size();
      //   //fmt::printf("index %d %d (size %d) has %d chunks  (sum %d)\n", index, subindex, getSizeFor(index, subindex), spans[i].size(), sum);
      // }
    }
    fmt::printf(" total %d chunks, sum %dM\n", nChunks, sum / 1024 / 1024);
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
    //allocator.debugInfo();

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
    //memsize += 1024 * 1024 * 2;
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
    std::lock_guard l(expandMutex);
    expand(size);
    return allocator.allocate(size);
  }
  std::pair<void*, size_t> allocate(size_t size) {
    //return {std::malloc(size), size};
    auto r = allocator.allocate(size);
    if (r.first) {
      [[likely]];
      return r;
    } else {
      return expandAndAllocate(size);
    }
  }
  void deallocate(void* ptr, size_t size) {
    //return std::free(ptr);
    //fmt::printf("deallocate [%#x, %#x)\n", (uintptr_t)ptr, (uintptr_t)ptr + size);
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

void MemfdAllocator::debugInfo() {
  return impl->allocator.debugInfo();
}

}

}
