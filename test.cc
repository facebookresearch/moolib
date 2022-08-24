#include "src/hash_map.h"

namespace h2 {
#include "src/hash_map2.h"
}

namespace h3 {
#include "src/hash_map3.h"
}

namespace h4 {
#include "src/hash_map4.h"
}

namespace h5 {
#include "src/hash_map5.h"
}

#include "../tmp/hash-table-shootout/flat_hash_map/flat_hash_map.hpp"

#include "src/memory/memfd.h"

#include "src/memory/buffer.h"

extern "C" {
#include "src/memory/rpmalloc.h"
}

#include "../mimalloc/include/mimalloc.h"

#include "fmt/printf.h"

#include <string>
#include <unordered_map>
#include <chrono>
#include <cassert>
#include <vector>
#include <atomic>
#include <mutex>
#include <random>
#include <thread>

#include <x86intrin.h>

template<typename Duration>
float seconds(Duration duration) {
  return std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(duration).count();
}

uint64_t times[1000];
const char* names[1000];
uint64_t counts[1000];

inline struct Timekeeper {
  int64_t tscThreshold = 0;
  int64_t tscDivisor = 0;
  int64_t prevTime = 0;
  int64_t prevTimeTsc = 0;
  int64_t lastBenchmarkTime = 0;
  int64_t lastBenchmarkTsc = 0;
  std::atomic_int64_t count = 0;
  std::mutex mutex;
  int64_t longPath(int64_t tsc) {
    std::unique_lock l(mutex, std::try_to_lock);
    if (!l.owns_lock()) {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    prevTime = now;
    prevTimeTsc = tsc;
    const int64_t benchmarkInterval = 1000000000;
    const int64_t resetTimeInterval = 100000000;
    if (now - lastBenchmarkTime >= (tscThreshold ? benchmarkInterval : benchmarkInterval / 10)) {
      int64_t pt = lastBenchmarkTime;
      int64_t pttsc = lastBenchmarkTsc;
      lastBenchmarkTime = now;
      lastBenchmarkTsc = tsc;
      if (pttsc) {
        //fmt::printf("tsc passed in %d: %d\n", now - pt, tsc - pttsc);
        int64_t tscDiff = tsc - pttsc;
        int64_t timeDiff = now - pt;
        if (tscDiff > 0 && timeDiff > 0) {
          uint64_t a = tscDiff;
          uint64_t b = timeDiff;
          uint64_t x = (a << 16) / b;
          //fmt::printf("x is %#x\n", x);
          tscDivisor = x;
          //tscThreshold = (a * (b >> 16)) / (resetTimeInterval >> 16);
          tscThreshold = tscDivisor * resetTimeInterval / 65536;
          //fmt::printf("tscThreshold is now %d\n", tscThreshold);
        }
      }
    }
    return now;
  }
  int64_t now() {
    int64_t tsc = __rdtsc();
    int64_t tscPassed = tsc - prevTimeTsc;
    if (tscThreshold > 0 & tscPassed < tscThreshold) {
      //++count;
      //fmt::printf("tscPassed is %d, divisor %d\n", tscPassed, tscDivisor);
      return prevTime + (tscPassed * 65536) / tscDivisor;
    }
    return longPath(tsc);
  }
} timekeeper;

struct Clockz {
  using duration = std::chrono::nanoseconds;
  using rep = duration::rep;
  using period = duration::period;
  using time_point = std::chrono::time_point<Clockz, duration>;
  static constexpr bool is_steady = true;

  static time_point now() noexcept {
    auto r = time_point(std::chrono::nanoseconds(timekeeper.now()));
    // auto now = std::chrono::steady_clock::now();
    // auto error = now.time_since_epoch() - r.time_since_epoch();
    // thread_local int count = 0;
    // if (++count % 100 == 0) {
    //   fmt::printf("thread %d, tsc %d, %d vs %d error is %d (%gms)\n", ::gettid(), __rdtsc(), r.time_since_epoch().count(), now.time_since_epoch().count(), error.count(), seconds(error) * 1000);
    // }
    return r;
  }
};

using Clock = std::chrono::steady_clock;

//#define time(name, x) start = __rdtsc(); std::atomic_thread_fence(std::memory_order_seq_cst); x; std::atomic_thread_fence(std::memory_order_seq_cst); end = __rdtsc(); names[__LINE__] = #name; times[__LINE__] += end - start; counts[__LINE__] += 1;
//#define time(name, x) start = __rdtsc(); x; end = __rdtsc(); names[__LINE__] = #name; times[__LINE__] += end - start; counts[__LINE__] += 1;
#define time(name, x) x

#define unrollSetBits(inputvalue, code) \
if (inputvalue) {uint64_t value__ = inputvalue; int index = __builtin_ctzll(value__); value__ >>= index + 1;{code}\
if (value__) {size_t index__ = __builtin_ctzll(value__) + 1; index += index__; value__ >>= index__;{code}\
if (value__) {size_t index__ = __builtin_ctzll(value__) + 1; index += index__; value__ >>= index__;{code}\
if (value__) {size_t index__ = __builtin_ctzll(value__) + 1; index += index__; value__ >>= index__;{code}\
if (value__) {size_t index__ = __builtin_ctzll(value__) + 1; index += index__; value__ >>= index__;{code}\
if (value__) {size_t index__ = __builtin_ctzll(value__) + 1; index += index__; value__ >>= index__;{code}\
if (value__) {size_t index__ = __builtin_ctzll(value__) + 1; index += index__; value__ >>= index__;{code}\
if (value__) {size_t index__ = __builtin_ctzll(value__) + 1; index += index__; value__ >>= index__;{code}\
for (int i = 14; i; --i) {\
if (!value__) break; {size_t index__ = __builtin_ctzll(value__) + 1; index += index__; value__ >>= index__;{code}}\
if (!value__) break; {size_t index__ = __builtin_ctzll(value__) + 1; index += index__; value__ >>= index__;{code}}\
if (!value__) break; {size_t index__ = __builtin_ctzll(value__) + 1; index += index__; value__ >>= index__;{code}}\
if (!value__) break; {size_t index__ = __builtin_ctzll(value__) + 1; index += index__; value__ >>= index__;{code}}\
}\
}}}}}}}}

int main() {

  uint64_t value = 0x240000000003099;

  uint64_t recon = 0;

  rpc::memfd::MemfdAllocator memfdAllocator;

  {
    rpmalloc_initialize();

    auto allocate = [&](auto& v) {
      //v.first = mi_malloc(v.second);
      //v.first = std::malloc(v.second);
      v = memfdAllocator.allocate(v.second);
      //v.first = rpmalloc(v.second);
      // auto h = rpc::makeBuffer(v.second, 0);
      // v.first = h->data();
      // v.second = (uintptr_t)h.release();
    };
    auto deallocate = [&](auto& v) {
      //mi_free(v.first);
      //rpc::BufferHandle{(rpc::Buffer*)v.second};
      //std::free(v.first);
      memfdAllocator.deallocate(v.first, v.second);
      //rpfree(v.first);
    };

    //std::unordered_map<uintptr_t, size_t> map;

    auto func = [&](size_t index) {
      rpmalloc_thread_initialize();
      std::minstd_rand rng(42 + 420 * index);
      std::vector<std::pair<void*, size_t>> allocations;
      allocations.resize(200);

      for (int x = 0; x != 8; ++x) {
        for (int j = 0; j != 2; ++j) {

          auto start = Clock::now();
          int alloccount = 0;
          int freecount = 0;

          for (int i = 0; i != 400000; ++i) {
            size_t index = std::uniform_int_distribution<size_t>(0, allocations.size() - 1)(rng);
            auto& v = allocations[index];
            if (v.first) {
              // map.erase((uintptr_t)v.first);
              // char val = *((char*)v.first);
              // for (size_t i = 0; i != 16 << j; ++i) {
              //   if (((char*)v.first)[i] != val) {
              //     fmt::printf("val mismatch\n");
              //   }
              // }
              //fmt::printf("val %d ok\n", val);
              deallocate(v);
              v.first = nullptr;
              ++freecount;
            } else {
              size_t s = std::uniform_int_distribution<size_t>(1, 4096)(rng);
              //size_t s = std::uniform_int_distribution<size_t>(1, 4096 + 1024 * 1024 * j)(rng);
              //size_t s = 1024 * 1024 * 2 + std::uniform_int_distribution<size_t>(1, 4096 + 1024 * 1024 * j)(rng);
              //size_t s = std::uniform_int_distribution<size_t>(1, 32 << j)(rng);
              v.second = s;
              allocate(v);
              std::memset(v.first, i, std::min(s, (size_t)1024 * 128));
              //std::memset(v.first, i, 16);
              // uintptr_t address = (uintptr_t)v.first;
              // size_t size = s;
              // for (auto& [base, s] : map) {
              //   if (address < base + s && base < address + size) {
              //     fmt::printf("allocation at %#x %d overlaps with existing allocation at %#x %d\n", address, size, base, s);
              //     std::abort();
              //   }
              // }
              // map[address] = size;
              ++alloccount;
            }
          }

          fmt::printf("thread %d: %d allocations, %d frees\n", index, alloccount, freecount);

          fmt::printf("thread %d: took %g\n", index, seconds(Clock::now() - start));

          // start = Clock::now();
          freecount = 0;
          for (auto& v : allocations) {
            if (v.first) {
              //map.erase((uintptr_t)v.first);
              deallocate(v);
              v.first = nullptr;
              ++freecount;
            }
          }
          // fmt::printf("deallocated %d in %g\n", freecount, seconds(Clock::now() - start));

          // memfdAllocator.debugInfo();
        }
      }
    };

    //func(0);

    std::vector<std::thread> threads;
    for (size_t i = 0; i != 1; ++i) {
      threads.emplace_back([&, i]{
        func(i);
      });
    }

    for (auto& v : threads) {
      v.join();
    }

  }

  return 0;

  // for (int i = 0; i != 10000000; ++i) {
  //   //uint64_t value = rand() | ((uint64_t)rand() << 32);
  //   uint64_t value = std::uniform_int_distribution<uint64_t>(0, std::numeric_limits<uint64_t>::max())(rng);
  //   if (rand() % 16) {
  //     value = 0;
  //     while (rand() % 2 == 0) {
  //       value |= 1ull << (rand() % 64);
  //     }
  //   }
  //   uint64_t recon = 0;
  //   unrollSetBits(value, recon |= 1ull << index;);
  //   if (value != recon) {
  //     printf("%#lx %#lx\n", value, recon);
  //     std::abort();
  //   }
  // }

  // printf("ok\n");
  // printf("recon %#lx value %#lx   xor %#lx\n", recon, value, recon ^ value);

  // return 0;

  moolib::HashMap<int, int, std::hash<int>> map;
  //std::unordered_map<int, int> map;
  std::unordered_map<int, int> map2;
  //h2::moolib::HashMap<int, int> map;
  //h3::moolib::HashMap<int, int> map2;
  //ska::flat_hash_map<int, int, ska::power_of_two_std_hash<int64_t>> map2;
  //ska::flat_hash_map<int, int, ska::power_of_two_std_hash<int64_t>> map;

  std::vector<int> values;
  for (int i = 0; i != 2200000; ++i) {
//    values.push_back(rand());
//    srand(Clock::now().time_since_epoch().count());
  }

  // for (auto& v : values) {
  //   map.insert(v, 1);
  // }

  auto start = __rdtsc();
  // for (auto& v : values) {
  //   map.insert(v, 1);
  // }
  auto end = __rdtsc();
  // printf("full insert took %g\n", seconds(end - start));

  // int hits = 0;
  // int misses = 0;
  // for (auto i = map.begin(); i != map.end(); ++i) {
  //   if (i.vi == -1) {
  //     ++hits;
  //   } else {
  //     ++misses;
  //   }
  // }
  // printf("hits: %d misses: %d\n", hits, misses);

  //return 0;

    for (int j = 0; j != 4; ++j) {

    //   for (int i = 0; i != 10000000; ++i) {
    //     int k = rand();

    //     if (rand() % 2 == 0) {
    //       //printf("erase %d\n", k);
    //       time(map.find, auto i = map.find(k));
    //       time(map2.find, auto i2 = map2.find(k));
          
    //       //printf("%d\n", i != map.end());
    //       assert((i == map.end()) == (i2 == map2.end()));
    //       if (i != map.end()) {
    //         assert(i->first == i2->first);
    //         assert(i->second == i2->second);
    //         time(map.erase, map.erase(i));
    //         time(map2.erase, map2.erase(i2));
    //       }
    //     } else {
    //       //printf("add %d\n", k);
    //       time(map.try_emplace, auto i = map.try_emplace(k, k));
    //       time(map2.try_emplace, auto i2 = map2.try_emplace(k, k));
    //       // if (i.second != i2.second) {
    //       //   auto e = map.end();
    //       //   for (auto i = map.begin(); i != e; ++i) {
    //       //     printf("%ld %ld   -> %s\n", i.ki, i.vi, i->c_str());
    //       //   }
    //       // }
    //       assert(i.second == i2.second);
    //     }
    //     assert(map.size() == map2.size());

    //     // auto e = map.end();
    //     // for (auto i = map.begin(); i != e; ++i) {
    //     //   printf("%ld %ld   -> %d\n", i.ki, i.vi, *i);
    //     // }

        

    // }

    // std::unordered_map<int, int> map3;
    // for (auto& [k, v] : map) {
    //   map3.emplace(v, v);
    // }
    // if (map2 != map3) {
    //   auto i = map2.begin();
    //   auto i2 = map3.begin();
    //   while (true) {
    //     if (i == map2.end() || i2 == map3.end()) {
    //       break;
    //     }
    //     printf("%d, %d  vs %d, %d\n", i->first, i->second, i2->first, i2->second);
    //     ++i;
    //     ++i2;
    //   }
    // }
    // assert(map3 == map2);
    // printf("ok\n");

    //constexpr int mask = 16777215;
    constexpr uint64_t mask = 0xffffffff;

    //uint32_t seed = Clock::now().time_since_epoch().count();
    uint32_t seed = 0x222d95ca;
    printf("seed %#x\n", seed);
    srand(seed);

    std::vector<int> ids;
    for (int i = 0; i != 10000000; ++i) {
      //ids.push_back((i + 1) & 0xffff);
      ids.push_back(rand());
    }

    std::random_shuffle(ids.begin(), ids.end());

    auto ids2 = ids;
    //std::random_shuffle(ids2.begin(), ids2.end());
    for (auto i = ids2.begin(); i != ids2.end(); ++i) {
      if (ids2.end() - i >= 10000) {
        std::random_shuffle(i, i + 500);
        i += rand() % 200;
      }
    }

    // std::vector<int> dummyids;
    // for (int i = 0; i != 10000000; ++i) {
    //   dummyids.push_back(i);
    // }
    // std::random_shuffle(dummyids.begin(), dummyids.end());

    auto start1 = Clock::now();
    auto start1c = std::chrono::steady_clock::now();
    {
      size_t maxsize = 0;
      auto i2 = ids2.begin();
      int n = 0;
      for (auto i = ids.begin(); i != ids.end(); ++i) {
        time(map.insert, map.emplace(*i, *i));
        maxsize = std::max(maxsize, map.size());
        while (i2 != ids2.end()) {
          auto i = map.find(*i2);
          if (i != map.end()) {
            map.erase(i);
            ++i2;
          } else {
            break;
          }
        }
        // ++n;
        // if (i > i2 && rand() % 3 != 0) {
        //   assert(map.find(*i2) != map.end());
        //   time(map.erase, map.erase(*i2));
        //   ++i2;
        // }
      }
      fmt::printf("maxsize %d\n", maxsize);
    }
    assert(map.size() == std::distance(map.begin(), map.end()));
    auto end1 = Clock::now();
    auto end1c = std::chrono::steady_clock::now();

    // auto* groups = map.groups;
    // auto* end = map.indexGroup(groups, map.ksize);
    // for (auto* i = groups; i != end; i = map.nextGroup(i)) {
    //   uint32_t distances = i->ext & 0xfffffff;
    //   uint8_t maxDistance = i->ext >> 28;
    //   printf("group %d has distances %#x (max %d)\n", map.groupIndex(groups, i), distances, maxDistance);
    // }

    // for (auto i = map.begin(); i != map.end(); ++i) {
    //   printf("  %d %d -> %d\n", i.gi, i.vi, i->first, i->second);
    // }

    // // dummyids.clear();
    // // for (int i = 0; i != 10000000; ++i) {
    // //   dummyids.push_back(i);
    // // }
    // // std::random_shuffle(dummyids.begin(), dummyids.end());

    // auto start2 = Clock::now();
    // auto start2c = std::chrono::steady_clock::now();
    // {
    //   auto i2 = ids.begin();
    //   int n = 0;
    //   for (auto i = ids.begin(); i != ids.end(); ++i) {
    //     time(map2.insert, map2.emplace(*i, *i));
    //     ++n;
    //     if (n >= 100000) {
    //       assert(map2.find(*i2) != map2.end());
    //       time(map2.erase, map2.erase(*i2));
    //       ++i2;
    //     }
    //   }
    // }
    // auto end2 = Clock::now();
    // auto end2c = std::chrono::steady_clock::now();

    // auto start1 = Clock::now();
    // for (int i = 0; i != 10000000; ++i) {
    //   int k = rand() % mask;
    //   if (rand() % 2 == 0) {
    //     time(map.find, auto i = map.find(k));
    //     if (i != map.end()) {
    //       time(map.erase, i = map.erase(i));
    //     }
    //   } else {
    //     time(map.try_emplace, auto i = map.try_emplace(k, k));
    //   }
    // }
    // auto end1 = Clock::now();

    // auto start2 = Clock::now();
    // for (int i = 0; i != 10000000; ++i) {
    //   int k = rand() % mask;
    //   if (rand() % 2 == 0) {
    //     time(map2.find, auto i = map2.find(k));
    //     if (i != map2.end()) {
    //       time(map2.erase, i = map2.erase(i));
    //     }
    //   } else {
    //     time(map2.try_emplace, auto i = map2.emplace(k, k));
    //   }
    // }
    // auto end2 = Clock::now();

    // std::unordered_map<int, int> vmap1;
    // for (auto& v : map) {
    //   vmap1[v] = v;
    // }
    // std::unordered_map<int, int> vmap2;
    // for (auto& v : map2) {
    //   vmap2[v.second] = v.second;
    // }

    // assert(vmap1 == vmap2);

    for (size_t i = 0; i != 1000; ++i) {
      if (names[i]) {
        printf("%s took %g\n", names[i], times[i] / (double)counts[i]);
      }
    }

    printf("total time:  map: %gms\n", seconds(end1 - start1) * 1000);
    // printf("total times:  map: %gms  map2: %gms\n", seconds(end1 - start1) * 1000, seconds(end2 - start2) * 1000);
    // printf("total times c:  map: %gms  map2: %gms\n", seconds(end1c - start1c) * 1000, seconds(end2c - start2c) * 1000);

    printf("map bucket count %ld  size %ld\n", map.bucket_count(), map.size());
    // printf("map2 bucket count %ld  size %ld\n", map2.bucket_count(), map2.size());

    // int hits = 0;
    // int misses = 0;
    // for (auto i = map.begin(); i != map.end(); ++i) {
    //   if (i.vi == -1) {
    //     ++hits;
    //   } else {
    //     ++misses;
    //   }
    // }
    // printf("map hits: %d misses: %d\n", hits, misses);

    // hits = 0;
    // misses = 0;
    // for (auto i = map2.begin(); i != map2.end(); ++i) {
    //   if (i.vi == -1) {
    //     ++hits;
    //   } else {
    //     ++misses;
    //   }
    // }
    // printf("map2 hits: %d misses: %d\n", hits, misses);

    printf("\n");
    map.clear();
    map2.clear();
    memset(names, 0, sizeof(names));
    memset(times, 0, sizeof(times));
    memset(counts, 0, sizeof(counts));
  }

  return 0;
}