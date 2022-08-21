
#include "memfd.h"
#include "synchronization.h"

#include "fmt/printf.h"

#include <vector>
#include <algorithm>
#include <utility>
#include <system_error>
#include <memory>

#include <unistd.h>
#include <sys/mman.h>

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

struct Span {
  uintptr_t begin;
  uintptr_t end;
};

struct Size {
  size_t size;
  uintptr_t begin;
};

static Size spanInsert(std::vector<Span>& spans, Span s) {
  auto i = std::lower_bound(spans.begin(), spans.end(), s.begin, [](auto& a, uintptr_t b) {
    return a.begin < b;
  });
  Size ss;
  if (i != spans.begin() && std::prev(i)->end == s.begin) {
    if (i != spans.end() && i->begin == s.end) {
      ss.begin = std::prev(i)->begin;
      ss.size = i->end - ss.begin;
      i = std::prev(spans.erase(i));
      i->begin = ss.begin;
      i->end = ss.begin + ss.size;
    } else {
      i = std::prev(i);
      i->end = s.end;
      ss.size = i->end - i->begin;
      ss.begin = i->begin;
    }
  } else if (i != spans.end() && i->begin == s.end) {
    i->begin = s.begin;
    ss.size = i->end - i->begin;
    ss.begin = i->begin;
  } else {
    spans.insert(i, s);
    ss.size = s.end - s.begin;
    ss.begin = s.begin;
  }
  return ss;
}

struct AllocatorImpl {
  std::vector<Span> spans;
  std::vector<Size> sizes;
  void addArea(void* base, size_t size) {
    Span s;
    s.begin = (uintptr_t)base;
    s.end = s.begin + size;
    insert(s);
  }
  std::pair<void*, size_t> allocate(size_t size) {
    size = (size + 63) / 64u * 64u;
    auto i = std::lower_bound(sizes.begin(), sizes.end(), size, [](auto& a, size_t b) {
      return a.size < b;
    });
    auto ii = i;
    while (i != sizes.end()) {
      uintptr_t begin = i->begin;
      ++i;
      auto i2 = std::lower_bound(spans.begin(), spans.end(), begin, [](auto& a, uintptr_t b) {
        return a.begin < b;
      });
      if (i2 == spans.end() || i2->begin != begin || i2->end - i2->begin < size) {
        while (i != sizes.end() && i->begin == begin) {
          ++i;
        }
        continue;
      }
      sizes.erase(ii, i);
      size_t space = i2->end - i2->begin;
      if (space - size < std::min((size_t)4096, size / 2u)) {
        size = space;
        spans.erase(i2);
      } else {
        i2->begin += size;
        Size ss;
        ss.begin = i2->begin;
        ss.size = i2->end - i2->begin;
        sizes.insert(std::lower_bound(sizes.begin(), sizes.end(), ss.size, [](auto& a, size_t b) {
          return a.size < b;
        }), ss);
      }
      //fmt::printf("allocate -> %p, %d\n", (void*)begin, size);
      return {(void*)begin, size};
    }
    //fmt::printf("allocate failed\n");
    sizes.erase(ii, i);
    return {nullptr, 0};
  }
  void deallocate(void* ptr, size_t size) {
    //fmt::printf("deallocate %p, %d\n", ptr, size);
    addArea(ptr, size);
  }

  void insert(Span s) {
    Size ss = spanInsert(spans, s);
    sizes.insert(std::lower_bound(sizes.begin(), sizes.end(), ss.size, [](auto& a, size_t b) {
      return a.size < b;
    }), ss);
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
  std::pair<void*, size_t> allocate(size_t size) {
    std::lock_guard l(mutex);
    while (true) {
      auto r = allocator.allocate(size);
      if (!r.first) {
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
        fmt::printf("memfd create %d (%dM) bytes\n", memsize, memsize / 1024 / 1024);
        auto memfd = Memfd::create(memsize);
        void* base = memfd.base;
        allocator.addArea((char*)base + 64, memfd.size - 128);
        auto* ptr = &fdToMemfd[memfd.fd];
        *ptr = {};
        ptr->memfd = std::move(memfd);
        memfds.insert(std::lower_bound(memfds.begin(), memfds.end(), (uintptr_t)base, [](auto& a, uintptr_t b) {
          return (uintptr_t)a->memfd.base < b;
        }), ptr);
        allocated += memsize;
        fmt::printf("allocated -> %dM\n", allocated / 1024 / 1024);
      } else {
        //fmt::printf("allocate [%#x, %#x)\n", (uintptr_t)r.first, (uintptr_t)r.first + r.second);
        return r;
      }
    }
  }
  void deallocate(void* ptr, size_t size) {
    //fmt::printf("deallocate [%#x, %#x)\n", (uintptr_t)ptr, (uintptr_t)ptr + size);
    std::lock_guard l(mutex);
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

}

}