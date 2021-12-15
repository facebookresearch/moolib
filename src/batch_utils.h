/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>

namespace moolib {
namespace utils {

namespace py = pybind11;

// TODO: Merge these functions with Batcher.

py::object squeezeFields(const py::handle& input, int64_t dim);
py::object unsqueezeFields(const py::handle& input, int64_t dim);

py::object stackFields(const py::tuple& input, int64_t dim);
py::tuple unstackFields(const py::handle& input, int64_t batchSize, int64_t dim);

} // namespace utils
} // namespace moolib
