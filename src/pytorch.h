/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "serialization.h"
#include "tensor.h"

#include <torch/torch.h>

namespace rpc {

Tensor torchTensorToTensor(const torch::Tensor&);
torch::Tensor tensorToTorchTensor(Tensor&&);

template<typename X>
void serialize(X& x, const torch::Tensor& v) {
  serialize(x, torchTensorToTensor(v));
}

template<typename X>
void serialize(X& x, torch::Tensor& v) {
  v = tensorToTorchTensor(x.template read<Tensor>());
}

} // namespace rpc
