/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "async.h"
#include "synchronization.h"

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

struct Thread {
  alignas(64) rpc::Semaphore sem;
  FunctionPointer f;
  std::thread thread;
  int n = 0;
  template<typename WaitFunction>
  void entry(WaitFunction&& waitFunction) noexcept {
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
        maxThreads = n;
      }
    }
#endif
  }
  ThreadPool(size_t maxThreads) : maxThreads(maxThreads) {}
  ~ThreadPool() {
    for (auto& v : threads) {
      v.thread.join();
    }
  }
  template<typename WaitFunction>
  bool addThread(Function<void()>& f, WaitFunction&& waitFunction) noexcept {
    std::unique_lock l(mutex);
    if (numThreads >= maxThreads) {
      return false;
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
    return true;
  }
  void setMaxThreads(size_t n) {
    std::unique_lock l(mutex);
    maxThreads = n;
  }
};

struct SchedulerFifoImpl {
  rpc::SpinMutex mutex;
  bool terminate = false;
  std::vector<Thread*> idle;
  FunctionPointer queueBegin = nullptr;
  FunctionPointer queueEnd = nullptr;

  ThreadPool pool;

  SchedulerFifoImpl() = default;
  SchedulerFifoImpl(size_t nThreads) : pool(nThreads) {}

  ~SchedulerFifoImpl() {
    std::unique_lock l(mutex);
    terminate = true;
    for (auto& v : idle) {
      v->f = nullptr;
      v->sem.post();
    }
    idle.clear();
    for (auto i = queueBegin; i != queueEnd;) {
      auto x = i;
      i = i->next;
      (Function<void()>)(x);
    }
  }

  void wait(Thread* t) noexcept {
    std::unique_lock l(mutex);
    if (terminate) {
      return;
    }
    if (queueBegin) {
      t->f = queueBegin;
      if (queueEnd == queueBegin) {
        queueBegin = queueEnd = nullptr;
      } else {
        queueBegin = queueBegin->next;
      }
    } else {
      idle.push_back(t);
      l.unlock();
      t->sem.wait();
    }
  }

  void run(Function<void()> f) noexcept {
    std::unique_lock l(mutex);
    if (idle.empty()) {
      if (pool.numThreads.load(std::memory_order_relaxed) < pool.maxThreads) {
        if (pool.addThread(f, [this](Thread* t) { wait(t); })) {
          return;
        }
      }
      FunctionPointer p = f.release();
      p->next = nullptr;
      if (queueEnd) {
        queueEnd->next = p;
        queueEnd = p;
      } else {
        queueBegin = queueEnd = p;
      }
    } else {
      Thread* t = idle.back();
      idle.pop_back();
      l.unlock();
      t->f = f.release();
      t->sem.post();
    }
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
