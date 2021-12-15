/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "fmt/printf.h"
#include "pybind11/pybind11.h"

#include <cstdlib>
#include <ctime>
#include <mutex>

namespace moolib {

namespace py = pybind11;

inline py::object pyLogging;

inline std::mutex logMutex;

enum class LogLevel {
  LOG_NONE,
  LOG_ERROR,
  LOG_INFO,
  LOG_VERBOSE,
  LOG_DEBUG,
};

constexpr auto LOG_NONE = LogLevel::LOG_NONE;
constexpr auto LOG_ERROR = LogLevel::LOG_ERROR;
constexpr auto LOG_INFO = LogLevel::LOG_INFO;
constexpr auto LOG_VERBOSE = LogLevel::LOG_VERBOSE;
constexpr auto LOG_DEBUG = LogLevel::LOG_DEBUG;

inline LogLevel currentLogLevel = LOG_ERROR;

template<typename... Args>
void logat(LogLevel level, const char* fmt, Args&&... args) {
  if (level > currentLogLevel) {
    return;
  }
  if (!pyLogging || pyLogging.is_none()) {
    std::lock_guard l(logMutex);
    time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    auto* tm = std::localtime(&now);
    char buf[0x40];
    std::strftime(buf, sizeof(buf), "%d-%m-%Y %H:%M:%S", tm);
    auto s = fmt::sprintf(fmt, std::forward<Args>(args)...);
    if (!s.empty() && s.back() == '\n') {
      fmt::printf("%s: %s", buf, s);
    } else {
      fmt::printf("%s: %s\n", buf, s);
    }
    fflush(stdout);
    fflush(stderr);
  } else {
    auto s = fmt::sprintf(fmt, std::forward<Args>(args)...);
    if (s.size() && s.back() == '\n') {
      s.pop_back();
    }
    s = fmt::sprintf("%d: %s", getpid(), s);
    py::gil_scoped_acquire gil;
    if (level == LOG_ERROR) {
      pyLogging.attr("error")(s);
    } else if (level == LOG_DEBUG) {
      // pyLogging.attr("debug")(s);
      pyLogging.attr("info")(s);
    } else {
      pyLogging.attr("info")(s);
    }
  }
}

inline struct Log {
  template<typename... Args>
  void error(const char* fmt, Args&&... args) {
    logat(LOG_ERROR, fmt, std::forward<Args>(args)...);
  }
  template<typename... Args>
  void info(const char* fmt, Args&&... args) {
    logat(LOG_INFO, fmt, std::forward<Args>(args)...);
  }
  template<typename... Args>
  void verbose(const char* fmt, Args&&... args) {
    logat(LOG_VERBOSE, fmt, std::forward<Args>(args)...);
  }
  template<typename... Args>
  void debug(const char* fmt, Args&&... args) {
    logat(LOG_DEBUG, fmt, std::forward<Args>(args)...);
  }
} log;

template<typename... Args>
[[noreturn]] void fatal(const char* fmt, Args&&... args) {
  auto s = fmt::sprintf(fmt, std::forward<Args>(args)...);
  log.error(" -- FATAL ERROR --\n%s\n", s);
  std::abort();
}

} // namespace moolib
