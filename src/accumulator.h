/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "pybind11/pybind11.h"

#include "rpc.h"
#include "util.h"

namespace moolib {

namespace py = pybind11;

struct Group;

struct AccumulatorImpl;

struct GradientStats {
  int numGradients = 0;
  int numSkipped = 0;
  int batchSize = 0;
};

struct Accumulator {

  std::unique_ptr<AccumulatorImpl> impl;

  Accumulator(std::string name, py::object parameters, py::object buffers, const Group* group = nullptr);
  ~Accumulator();

  void update();

  void connect(std::string address);

  bool connected();
  bool wantsState();
  bool hasNewState();
  bool hasGradients();
  bool wantsGradients();
  void setState(py::object userState);
  py::object state();

  void skipGradients();
  void reduceGradients(int batchSize);
  void zeroGradients();

  int64_t modelVersion() const;
  void setModelVersion(int64_t n);
  py::dict getGradientStats() const;

  void setVirtualBatchSize(int n);
  void setParallelGradients(int n);

  std::string getLeader();
  bool isLeader();
};

} // namespace moolib
