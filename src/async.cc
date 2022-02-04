/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "async.h"
#include "synchronization.h"

#include <algorithm>
#include <atomic>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace async {

void setCurrentThreadName(const std::string& name) {
#ifdef __APPLE__
  pthread_setname_np(name.c_str());
#elif __linux__
  pthread_setname_np(pthread_self(), name.c_str());
#endif
}

template<typename T>
using Function = rpc::Function<T>;
using FunctionPointer = rpc::FunctionPointer;

thread_local bool isThisAThread = false;

struct Thread {
  alignas(64) rpc::Semaphore sem;
  FunctionPointer f = nullptr;
  std::thread thread;
  int n = 0;
  std::atomic_bool terminate = false;
  template<typename WaitFunction>
  void entry(WaitFunction&& waitFunction) noexcept {
    isThisAThread = true;
    while (f) {
      try {
        Function<void()>{f}();
      } catch (const std::exception& e) {
        fprintf(stderr, "Unhandled exception in async function: %s\n", e.what());
        fflush(stderr);
        std::abort();
      }
      f = nullptr;
      waitFunction(this);
    }
  }
};

struct ThreadPool {
  alignas(64) rpc::SpinMutex mutex;
  std::list<Thread> threads;
  std::atomic_size_t numThreads = 0;
  size_t maxThreads = 0;
  ThreadPool() noexcept {
    maxThreads = std::thread::hardware_concurrency();
    if (maxThreads < 1) {
      maxThreads = 1;
    }
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
      int n = CPU_COUNT(&set);
      if (n > 0) {
        maxThreads = std::max(n - 1, 1);
      }
    }
#endif
    maxThreads = std::min(maxThreads, (size_t)64);
  }
  ThreadPool(size_t maxThreads) : maxThreads(std::min(maxThreads, (size_t)64)) {}
  ~ThreadPool() {
    for (auto& v : threads) {
      v.thread.join();
    }
  }
  template<typename WaitFunction>
  Thread* addThread(Function<void()>& f, WaitFunction&& waitFunction) noexcept {
    std::unique_lock l(mutex);
    if (numThreads >= maxThreads) {
      return nullptr;
    }
    stopForksFromHereOn();
    int n = ++numThreads;
    threads.emplace_back();
    auto* t = &threads.back();
    t->n = threads.size() - 1;
    t->f = f.release();
    rpc::Semaphore sem;
    std::atomic<bool> started = false;
    t->thread = std::thread([&sem, &started, n, t, waitFunction = std::forward<WaitFunction>(waitFunction)]() mutable {
      setCurrentThreadName("async " + std::to_string(n));
      started = true;
      sem.post();
      t->entry(std::move(waitFunction));
    });
    sem.wait();
    return t;
  }
  void setMaxThreads(size_t n) {
    std::unique_lock l(mutex);
    maxThreads = n;
  }
};

struct SchedulerFifoImpl {
  std::array<std::atomic<Thread*>, 64> threads{};
  std::atomic_size_t nextThreadIndex = 0;

  struct alignas(64) Incoming {
    rpc::SpinMutex mutex;
    std::atomic_size_t req = 0;
    std::array<FunctionPointer, 8> p{};
  };
  struct alignas(64) Outgoing {
    std::atomic<rpc::Semaphore*> sleeping = nullptr;
    size_t ack = 0;
  };
  std::array<Incoming, 64> incoming;
  std::array<Outgoing, 64> outgoing;

  std::atomic_size_t globalQueueSize = 0;
  rpc::SpinMutex globalMutex;
  size_t globalQueueOffset = 0;
  std::vector<rpc::Function<void()>> globalQueue;

  ThreadPool pool;

  SchedulerFifoImpl() = default;
  SchedulerFifoImpl(size_t nThreads) : pool(nThreads) {}

  ~SchedulerFifoImpl() {
    for (auto& v : threads) {
      Thread* t = v.load();
      if (t) {
        t->terminate = true;
      }
    }
  }

  std::atomic_int numIdleThreads = 0;

  void wait(Thread* t) noexcept {
    size_t index = t->n;
    auto& i = incoming[index];
    auto& o = outgoing[index];
    size_t ack = o.ack;
    size_t spinCount = 0;
    bool sleeping = false;
    while (i.req.load(std::memory_order_acquire) == ack) {
      _mm_pause();
      if (++spinCount >= 1 << 12) {
        if (t->terminate.load(std::memory_order_relaxed)) {
          return;
        }
        std::this_thread::yield();
        if (i.req.load(std::memory_order_acquire) != ack) {
          break;
        }
        if (spinCount >= (1 << 12) + 256) {
          int ms = sleeping ? 5 : 0;
          sleeping = true;
          o.sleeping.store(&t->sem, std::memory_order_relaxed);
          t->sem.wait_for(std::chrono::milliseconds(ms));
        }
      }

      if (globalQueueSize.load(std::memory_order_relaxed) != 0) {
        std::lock_guard l(globalMutex);
        if (globalQueueOffset != globalQueue.size()) {
          t->f = globalQueue[globalQueueOffset].release();
          ++globalQueueOffset;
          globalQueueSize.fetch_sub(1, std::memory_order_relaxed);
          if (globalQueueOffset >= 0x100) {
            globalQueue.erase(globalQueue.begin(), globalQueue.begin() + globalQueueOffset);
            globalQueueOffset = 0;
          }
          return;
        }
      }
    }
    o.sleeping.store(nullptr, std::memory_order_relaxed);
    t->f = i.p[ack % i.p.size()];
    ++o.ack;
  }

  void scheduleGlobally(Function<void()> f) {
    std::lock_guard l(globalMutex);
    globalQueueSize.fetch_add(1, std::memory_order_relaxed);
    globalQueue.push_back(std::move(f));
  }

  void run(Function<void()> f) noexcept {
    thread_local size_t tlsSearchOffset = 0;
    thread_local bool searchDirection = false;
    bool searchDir = searchDirection ^= true;
    size_t searchOffset = tlsSearchOffset;

    size_t nThreads = nextThreadIndex.load(std::memory_order_acquire);
    for (size_t n = 0; n != nThreads; ++n) {
      if (searchDir) {
        if (searchOffset == 0) {
          searchOffset = nThreads - 1;
        } else {
          --searchOffset;
        }
      } else {
        if (searchOffset == nThreads - 1) {
          searchOffset = 0;
        } else {
          ++searchOffset;
        }
      }
      auto& i = incoming[searchOffset];
      auto& o = outgoing[searchOffset];
      size_t ack = o.ack;
      if (i.req.load(std::memory_order_relaxed) == ack) {
        std::lock_guard l(i.mutex);
        size_t req = i.req.load(std::memory_order_relaxed);
        if (req == ack) {
          i.p[req % i.p.size()] = f.release();
          ++req;
          i.req.store(req, std::memory_order_relaxed);
          auto* sleeping = o.sleeping.load(std::memory_order_relaxed);
          if (sleeping) {
            sleeping->post();
          }
          return;
        }
      }
    }

    if (pool.numThreads.load(std::memory_order_relaxed) < pool.maxThreads) {
      if (Thread* t = pool.addThread(f, [this](Thread* t) { wait(t); })) {
        size_t index = nextThreadIndex++;
        if (index < threads.size()) {
          threads[index] = t;
        }
        return;
      }
    }

    nThreads = nextThreadIndex.load(std::memory_order_acquire);
    size_t bestIndex = -1;
    size_t bestOccupancy = 100;
    for (size_t n = 0; n != nThreads; ++n) {
      if (searchDir) {
        if (searchOffset == 0) {
          searchOffset = nThreads - 1;
        } else {
          --searchOffset;
        }
      } else {
        if (searchOffset == nThreads - 1) {
          searchOffset = 0;
        } else {
          ++searchOffset;
        }
      }
      auto& i = incoming[searchOffset];
      auto& o = outgoing[searchOffset];
      size_t ack = o.ack;
      size_t occupancy = i.req.load(std::memory_order_relaxed) - ack;
      if (occupancy < bestOccupancy) {
        bestOccupancy = occupancy;
        bestIndex = searchOffset;
        if (occupancy == 1) {
          break;
        }
      }
    }

    if (bestIndex != (size_t)-1) {
      auto& i = incoming[bestIndex];
      auto& o = outgoing[bestIndex];
      std::lock_guard l(i.mutex);
      size_t ack = o.ack;
      size_t req = i.req.load(std::memory_order_relaxed);
      if (req - ack < i.p.size()) {
        i.p[req % i.p.size()] = f.release();
        ++req;
        i.req.store(req, std::memory_order_relaxed);
        auto* sleeping = o.sleeping.load(std::memory_order_relaxed);
        if (sleeping) {
          sleeping->post();
        }
        return;
      }
    }
    scheduleGlobally(std::move(f));
  }
};

SchedulerFifo::SchedulerFifo() {
  impl_ = std::make_unique<SchedulerFifoImpl>();
}
SchedulerFifo::SchedulerFifo(size_t nThreads) {
  impl_ = std::make_unique<SchedulerFifoImpl>(nThreads);
}
SchedulerFifo::~SchedulerFifo() {}
void SchedulerFifo::run(Function<void()> f) noexcept {
  impl_->run(std::move(f));
}
void SchedulerFifo::setMaxThreads(size_t nThreads) {
  impl_->pool.setMaxThreads(nThreads);
}

bool SchedulerFifo::isInThread() const noexcept {
  return isThisAThread;
}

std::atomic_bool atForkHandlerInstalled = false;

void stopForksFromHereOn() {
  if (atForkHandlerInstalled || atForkHandlerInstalled.exchange(true)) {
    return;
  }
  auto* allowFork = getenv("MOOLIB_ALLOW_FORK");
  if (!allowFork || strlen(allowFork) == 0 || !strcmp(allowFork, "0")) {
    pthread_atfork(
        []() {
          fprintf(
              stderr, "Illegal fork detected! Moolib has already been initialized, so forking at this point is likely "
                      "to break things.\n"
                      "Please fork (including multiprocessing, moolib.EnvPool) before using moolib.Rpc (directly or "
                      "indirectly).\n"
                      "This check can be removed by setting the MOOLIB_ALLOW_FORK environment variable.\n");
          fflush(stderr);
          std::abort();
        },
        nullptr, nullptr);
  }
}

} // namespace async
