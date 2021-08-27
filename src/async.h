#pragma once

#include "function.h"

#include <memory>
#include <string>

namespace async {

void setCurrentThreadName(const std::string& name);

template<typename T>
using Function = rpc::Function<T>;
using FunctionPointer = rpc::FunctionPointer;

struct SchedulerFifoImpl;

struct SchedulerFifo {

  std::unique_ptr<SchedulerFifoImpl> impl_;

  SchedulerFifo();
  SchedulerFifo(size_t nThreads);
  ~SchedulerFifo();

  void run(Function<void()> f) noexcept;
  void setMaxThreads(size_t nThreads);
};

void stopForksFromHereOn();

} // namespace async
