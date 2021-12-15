/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "tensor.h"

#include <pybind11/pybind11.h>
#include <torch/python.h>
#include <torch/torch.h>

namespace rpc {

pybind11::object toPython(const Tensor& t) {
  return py::reinterpret_steal<py::object>(THPVariable_Wrap(t.impl.as<torch::Tensor>()));
}

std::optional<Tensor> tryFromPython(pybind11::handle v) {
  if (THPVariable_Check(v.ptr())) {
    Tensor r(nullptr);
    r.impl.emplace<torch::Tensor>(THPVariable_Unpack(v.ptr()));
    return r;
  } else {
    return {};
  }
}

void setPythonTensor(pybind11::handle o, const Tensor& t) {
  // pytorch used to return a non-const reference here.
  // Now it returns a const reference, but we really would like
  // to be able to assign to the internal tensor, and nothing
  // seems to catch on fire for our use case.
  const_cast<torch::Tensor&>(THPVariable_Unpack(o.ptr())) = t.impl.as<torch::Tensor>();
}

} // namespace rpc
