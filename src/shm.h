/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "logging.h"

#include <atomic>
#include <cstdlib>
#include <string>

#include <fcntl.h>

#ifndef __APPLE__
#include <semaphore.h>
#else
#include <pthread.h>
#endif
#include <sys/mman.h>

namespace moolib {

struct SharedMemory {
  int fd = -1;
  size_t size = 1024 * 1024 * 400;
  std::byte* data = nullptr;
  std::string name;
  bool unlinked = false;

  static_assert(std::atomic_bool::is_always_lock_free, "need lock-free atomics");
  struct InitBlock {
    std::atomic_bool initialized;
    std::atomic_bool initializing;
  };

  InitBlock* initBlock = nullptr;

  SharedMemory(std::string_view name) : name(name) {
    log.verbose("creating shm %s\n", name);
    fd = shm_open(std::string(name).c_str(), O_RDWR | O_CREAT, ACCESSPERMS);
    if (fd < 0) {
      throw std::system_error(errno, std::system_category(), "shm_open");
    }
    if (ftruncate(fd, size)) {
      /* Fails on OSX after the first time but can be ignored. */
      log.verbose("ftruncate failed on shm: %d\n", errno);
    }
    data = (std::byte*)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (!data) {
      throw std::system_error(errno, std::system_category(), "mmap");
    }
    initBlock = (InitBlock*)data;
    size_t n = (sizeof(InitBlock) + 63) / 64 * 64;
    data += n;
    size -= n;
  }
  ~SharedMemory() {
    munmap(data, size);
    close(fd);
    if (!unlinked) {
      shm_unlink(name.c_str());
    }
  }
  void unlink() {
    if (!unlinked) {
      shm_unlink(name.c_str());
      unlinked = true;
    }
  }
  template<typename T>
  T& as() {
    if (sizeof(T) > size) {
      fatal("%s is too big for shm :(\n", typeid(T).name());
    }
    T& r = *(T*)data;
    if (!initBlock->initialized) {
      if (initBlock->initializing.exchange(true)) {
        while (!initBlock->initialized)
          ;
        return r;
      }
      r.init(size);
      initBlock->initialized = true;
    }
    return r;
  }
};

class SharedSemaphore {
#ifndef __APPLE__
  sem_t sem;

public:
  SharedSemaphore() noexcept {
    sem_init(&sem, 1, 0);
  }

  ~SharedSemaphore() {
    sem_destroy(&sem);
  }

  void post() noexcept {
    sem_post(&sem);
  }

  void wait() noexcept {
    while (sem_wait(&sem)) {
      if (errno != EINTR) {
        std::abort();
      }
    }
  }

  template<typename Rep, typename Period>
  void wait_for(const std::chrono::duration<Rep, Period>& duration) noexcept {
    struct timespec ts;
    SharedSemaphore::fill_ts(ts, duration);
    while (sem_timedwait(&sem, &ts)) {
      if (errno == ETIMEDOUT) {
        break;
      }
      if (errno != EINTR) {
        std::abort();
      }
    }
  }

#else  /* __APPLE__ */
  pthread_mutex_t mu;
  pthread_cond_t cv;
  unsigned value;

  struct Lock {
    Lock(pthread_mutex_t* mutex) : mu(mutex) {
      int rc = pthread_mutex_lock(mu);
      if (rc) {
        fatal("pthread_mutex_lock: %d\n", rc);
      }
    }
    ~Lock() {
      int rc = pthread_mutex_unlock(mu);
      if (rc) {
        fatal("pthread_mutex_unlock: %d\n", rc);
      }
    }
    pthread_mutex_t* mu;
  };

public:
  SharedSemaphore() noexcept : value(0) {
    pthread_mutexattr_t psharedm;
    pthread_condattr_t psharedc;

    pthread_mutexattr_init(&psharedm);
    pthread_mutexattr_setpshared(&psharedm, PTHREAD_PROCESS_SHARED);
    pthread_condattr_init(&psharedc);
    pthread_condattr_setpshared(&psharedc, PTHREAD_PROCESS_SHARED);

    pthread_mutex_init(&mu, &psharedm);
    pthread_cond_init(&cv, &psharedc);
  }

  ~SharedSemaphore() {
    pthread_cond_destroy(&cv);
    pthread_mutex_destroy(&mu);
  }

  void post() noexcept {
    Lock lock(&mu);
    if (value == 0) {
      pthread_cond_signal(&cv);
    }
    value++;
  }

  void wait() noexcept {
    Lock lock(&mu);
    while (value == 0) {
      if (pthread_cond_wait(&cv, &mu)) {
        std::abort();
      }
    }
    --value;
  }

  template<typename Rep, typename Period>
  void wait_for(const std::chrono::duration<Rep, Period>& duration) noexcept {
    struct timespec ts;
    SharedSemaphore::fill_ts(ts, duration);

    Lock lock(&mu);

    int rc = 0;
    while (value == 0 && rc == 0) {
      rc = pthread_cond_timedwait(&cv, &mu, &ts);
    }

    if (rc == 0) {
      --value;
    } else if (rc != ETIMEDOUT) {
      std::abort();
    }
  }
#endif /* __APPLE__ */

  template<typename Clock, typename Duration>
  void wait_until(const std::chrono::time_point<Clock, Duration>& timePoint) noexcept {
    wait_for(timePoint - Clock::now());
  }

  SharedSemaphore(const SharedSemaphore&) = delete;
  SharedSemaphore(const SharedSemaphore&&) = delete;
  SharedSemaphore& operator=(const SharedSemaphore&) = delete;
  SharedSemaphore& operator=(const SharedSemaphore&&) = delete;

private:
  template<typename TimeSpec, typename Rep, typename Period>
  static void fill_ts(TimeSpec& ts, const std::chrono::duration<Rep, Period>& duration) {
    auto absduration = std::chrono::system_clock::now().time_since_epoch() + duration;
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(absduration);
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(nanoseconds);
    ts.tv_sec = seconds.count();
    ts.tv_nsec = (nanoseconds - seconds).count();
  }
};

template<typename T>
struct SharedPointer {
  size_t offset = 0;
  template<typename Shared>
  T* operator()(Shared* shared) {
    return (T*)((std::byte*)shared + offset);
  }
};

template<typename T>
struct SharedArray {
  size_t size;
  SharedPointer<T> data;

  template<typename Shared>
  std::basic_string_view<T> view(Shared* shared) {
    return {data(shared), size};
  }

  template<typename Shared>
  T& operator()(Shared* shared, size_t index) {
    return data(shared)[index];
  }
  template<typename Shared>
  T* operator()(Shared* shared) {
    return data(shared);
  }
};

} // namespace moolib
