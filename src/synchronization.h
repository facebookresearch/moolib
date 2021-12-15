/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <mutex>
#include <thread>

#ifdef __linux__
#include <semaphore.h>
#endif

#include <x86intrin.h>

namespace rpc {

#if 0
using SpinMutex = std::mutex;
#elif 0
inline std::atomic_int mutexThreadIdCounter = 0;
inline thread_local int mutexThreadId = mutexThreadIdCounter++;
class SpinMutex {
  int magic = 0x42;
  std::atomic<bool> locked_{false};
  std::atomic<int*> owner = nullptr;

public:
  void lock() {
    if (owner == &mutexThreadId) {
      printf("recursive lock\n");
      std::abort();
    }
    if (magic != 0x42) {
      printf("BAD MUTEX MAGIC\n");
      std::abort();
    }
    auto start = std::chrono::steady_clock::now();
    do {
      while (locked_.load(std::memory_order_acquire)) {
        _mm_pause();
        if (magic != 0x42) {
          printf("BAD MUTEX MAGIC\n");
          std::abort();
        }
        if (std::chrono::steady_clock::now() - start >= std::chrono::seconds(10)) {
          int* p = owner.load();
          printf("deadlock detected in thread %d! held by thread %d\n", mutexThreadId, p ? *p : -1);
          start = std::chrono::steady_clock::now();
        }
      }
    } while (locked_.exchange(true, std::memory_order_acquire));
    owner = &mutexThreadId;
  }
  void unlock() {
    if (magic != 0x42) {
      printf("BAD MUTEX MAGIC\n");
      std::abort();
    }
    owner = nullptr;
    locked_.store(false);
  }
  bool try_lock() {
    if (owner == &mutexThreadId) {
      printf("recursive try_lock\n");
      std::abort();
    }
    if (locked_.load(std::memory_order_acquire)) {
      return false;
    }
    bool r = !locked_.exchange(true, std::memory_order_acquire);
    if (r) {
      owner = &mutexThreadId;
    }
    return r;
  }
};
#else
class SpinMutex {
  std::atomic<bool> locked_{false};

public:
  void lock() {
    do {
      while (locked_.load(std::memory_order_acquire)) {
        _mm_pause();
      }
    } while (locked_.exchange(true, std::memory_order_acquire));
  }
  void unlock() {
    locked_.store(false);
  }
  bool try_lock() {
    if (locked_.load(std::memory_order_acquire)) {
      return false;
    }
    return !locked_.exchange(true, std::memory_order_acquire);
  }
};
#endif

#ifdef __linux__
class Semaphore {
  sem_t sem;

public:
  Semaphore() noexcept {
    sem_init(&sem, 0, 0);
  }
  ~Semaphore() {
    sem_destroy(&sem);
  }
  void post() noexcept {
    sem_post(&sem);
  }
  void wait() noexcept {
    while (sem_wait(&sem)) {
      if (errno != EINTR) {
        printf("sem_wait returned errno %d", (int)errno);
        std::abort();
      }
    }
  }
  template<typename Rep, typename Period>
  void wait_for(const std::chrono::duration<Rep, Period>& duration) noexcept {
    struct timespec ts;
    auto absduration = std::chrono::system_clock::now().time_since_epoch() + duration;
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(absduration);
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(nanoseconds);
    ts.tv_sec = seconds.count();
    ts.tv_nsec = (nanoseconds - seconds).count();
    while (sem_timedwait(&sem, &ts)) {
      if (errno == ETIMEDOUT) {
        break;
      }
      if (errno != EINTR) {
        printf("sem_timedwait returned errno %d", (int)errno);
        std::abort();
      }
    }
  }
  template<typename Clock, typename Duration>
  void wait_until(const std::chrono::time_point<Clock, Duration>& timePoint) noexcept {
    wait_for(timePoint - Clock::now());
  }

  Semaphore(const Semaphore&) = delete;
  Semaphore(const Semaphore&&) = delete;
  Semaphore& operator=(const Semaphore&) = delete;
  Semaphore& operator=(const Semaphore&&) = delete;
};
#else
class Semaphore {
  int count_ = 0;
  std::mutex mut_;
  std::condition_variable cv_;

public:
  void post() {
    std::unique_lock l(mut_);
    if (++count_ >= 1) {
      cv_.notify_one();
    }
  }
  void wait() {
    std::unique_lock l(mut_);
    while (count_ == 0) {
      cv_.wait(l);
    }
    --count_;
  }
  template<typename Clock, typename Duration>
  void wait_until(const std::chrono::time_point<Clock, Duration>& timePoint) noexcept {
    std::unique_lock l(mut_);
    while (count_ == 0) {
      if (cv_.wait_until(l, timePoint) == std::cv_status::timeout) return;
    }
    --count_;
  }

  template<typename Rep, typename Period>
  void wait_for(const std::chrono::duration<Rep, Period>& duration) noexcept {
    std::unique_lock l(mut_);
    if (cv_.wait_for(l, duration, [this]() { return count_ > 0; })) --count_;
  }
};
#endif

} // namespace rpc
