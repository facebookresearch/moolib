/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include "fmt/printf.h"

namespace {
struct Timer {
  std::chrono::steady_clock::time_point start;
  Timer() {
    reset();
  }
  void reset() {
    start = std::chrono::steady_clock::now();
  }
  float elapsedAt(std::chrono::steady_clock::time_point now) {
    return std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(now - start).count();
  }
  float elapsed() {
    return elapsedAt(std::chrono::steady_clock::now());
  }
  float elapsedReset() {
    auto now = std::chrono::steady_clock::now();
    float r = elapsedAt(now);
    start = now;
    return r;
  }
};

Timer mainTimer;
std::atomic<bool> isDone = false;

float timeoutSeconds = 300;

std::atomic<const char*> currentTest = "<none>";
std::atomic_int pass = 0;
std::atomic_int fail = 0;
std::atomic_int skip = 0;
std::mutex quitMutex;
std::thread timeoutThread;
std::once_flag timeoutOnce;

void quit() {
  if (isDone) {
    return;
  }
  {
    std::lock_guard l(quitMutex);
    if (isDone) {
      return;
    }
    fmt::printf("Passed: %d\nFailed: %d\nSkipped: %d\n", pass.load(), fail.load(), skip.load());
    fmt::printf("Ran %d tests in %gs\n", pass.load() + fail.load(), mainTimer.elapsed());
    fflush(stdout);
    isDone = true;
    if (timeoutThread.joinable()) {
      if (timeoutThread.get_id() != std::this_thread::get_id()) {
        timeoutThread.detach();
      } else {
        timeoutThread.join();
      }
    }
  }
  if (fail == 0) {
    std::exit(0);
  } else {
    std::quick_exit(1);
  }
}

void failAt(const char* file, int line, std::string str) {
  ++fail;
  fmt::printf("\n");
  fmt::printf("Test FAILED: %s at %s:%d\n", currentTest.load(), file, line);
  fmt::printf("Message: %s\n", str);
  fmt::printf("\n");
  fflush(stdout);
  quit();
}

void passAt(const char* file, int line, std::string str) {
  ++pass;
  fmt::printf("Passed: %s\n", currentTest.load());
}

#define FAIL(x) failAt(__FILE__, __LINE__, x)
#define PASS(x) passAt(__FILE__, __LINE__, x)
#define ASSERT(x)                                                                                                      \
  if (!(x)) FAIL(#x)

#define RUN(x, ...) runTest<x>(#x);
#define RUNARG(x, ...) runTest<x>(#x, __VA_ARGS__);

void startTimeoutThread() {
  std::call_once(timeoutOnce, [&] {
    timeoutThread = std::thread([&] {
      while (mainTimer.elapsed() < timeoutSeconds && !isDone) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      if (!isDone) {
        FAIL("Timed out waiting for tests to finish!");
      }
    });
  });
}

struct Test {
  Test() {}
  ~Test() {}
};

template<typename T, typename... Args>
void runTest(const char* name, Args&&... args) {
  startTimeoutThread();
  currentTest = name;
  T obj(std::forward<Args>(args)...);
  PASS("Passed");
}

} // namespace
