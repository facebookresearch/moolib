/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "batch_utils.h"

#include <algorithm>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include "pythonserialization.h"
#include "rpc.h"
#include "tensor.h"

namespace moolib {
namespace utils {

namespace {

template<class Function>
void visitNested(Function func, const py::handle& input) {
  if (py::isinstance<py::tuple>(input)) {
    const py::tuple& src = py::reinterpret_borrow<py::tuple>(input);
    for (const auto& x : src) {
      visitNested(func, x);
    }
    return;
  }

  if (py::isinstance<py::list>(input)) {
    const py::list& src = py::reinterpret_borrow<py::list>(input);
    for (const auto& x : src) {
      visitNested(func, x);
    }
    return;
  }

  if (py::isinstance<py::dict>(input)) {
    const py::dict& src = py::reinterpret_borrow<py::dict>(input);
    for (const auto& [k, v] : src) {
      visitNested(func, v);
    }
    return;
  }

  func(input);
}

template<class Function>
py::object mapNested(Function func, const py::handle& input) {
  if (py::isinstance<py::tuple>(input)) {
    const py::tuple& src = py::reinterpret_borrow<py::tuple>(input);
    const int64_t n = src.size();
    py::tuple dst(n);
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = mapNested(func, src[i]);
    }
    return dst;
  }

  if (py::isinstance<py::list>(input)) {
    const py::list& src = py::reinterpret_borrow<py::list>(input);
    const int64_t n = src.size();
    py::list dst(n);
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = mapNested(func, src[i]);
    }
    return dst;
  }

  if (py::isinstance<py::dict>(input)) {
    const py::dict& src = py::reinterpret_borrow<py::dict>(input);
    py::dict dst;
    for (const auto& [k, v] : src) {
      dst[k] = mapNested(func, v);
    }
    return dst;
  }

  return func(input);
}

// Implementation of squeezeFields.
// Returns a pair consisting a py::object which is the result and a bool denoting whether the result is a real node.
// If the child of a tuple node is not a real node, that means the current list is generated from squeezeFields and
// should do unsqueeze here.
std::pair<py::object, bool> squeezeFieldsImpl(const py::handle& input, int64_t dim) {
  if (py::isinstance<py::tuple>(input)) {
    const py::tuple& src = py::reinterpret_borrow<py::tuple>(input);
    const int64_t n = src.size();
    py::tuple dst(n);
    bool anyNode = false;
    for (int64_t i = 0; i < n; ++i) {
      auto [cur, tag] = squeezeFieldsImpl(src[i], dim);
      dst[i] = std::move(cur);
      anyNode |= tag;
    }
    if (n == 1 && !anyNode) {
      return std::make_pair(std::move(dst[0]), true);
    } else {
      return std::make_pair(std::move(dst), true);
    }
  }

  if (py::isinstance<py::list>(input)) {
    const py::list& src = py::reinterpret_borrow<py::list>(input);
    const int64_t n = src.size();
    py::list dst(n);
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = std::move(squeezeFieldsImpl(src[i], dim).first);
    }
    return std::make_pair(std::move(dst), true);
  }

  if (py::isinstance<py::dict>(input)) {
    const py::dict& src = py::reinterpret_borrow<py::dict>(input);
    py::dict dst;
    for (const auto& [k, v] : src) {
      dst[k] = std::move(squeezeFieldsImpl(v, dim).first);
    }
    return std::make_pair(std::move(dst), true);
  }

  const auto src = rpc::tryFromPython(input);
  if (src) {
    return std::make_pair(rpc::toPython(src->squeeze(dim)), true);
  } else {
    return std::make_pair(py::reinterpret_borrow<py::object>(input), false);
  }
}

// Prepare for unstackFields.
// batchTuple preserves the tag for each tuple node which indicates whether the tuple nodes should be unstacked.
// The items in the batchTuple are pushed in preorder traversal.
bool prepareForUnstack(const py::handle& input, std::vector<bool>& batchTuple) {
  if (py::isinstance<py::tuple>(input)) {
    const py::tuple& src = py::reinterpret_borrow<py::tuple>(input);
    const int64_t curIndex = batchTuple.size();
    batchTuple.push_back(false);
    bool anyNode = false;
    for (const auto& x : src) {
      anyNode |= prepareForUnstack(x, batchTuple);
    }
    batchTuple[curIndex] = !anyNode;
    return true;
  }
  if (py::isinstance<py::list>(input)) {
    const py::list& src = py::reinterpret_borrow<py::list>(input);
    for (const auto& x : src) {
      prepareForUnstack(x, batchTuple);
    }
    return true;
  }
  if (py::isinstance<py::dict>(input)) {
    const py::dict& src = py::reinterpret_borrow<py::dict>(input);
    for (const auto& [k, v] : src) {
      prepareForUnstack(v, batchTuple);
    }
    return true;
  }
  const auto src = rpc::tryFromPython(input);
  return src ? true : false;
}

template<class Sequence>
py::tuple unstackSequence(int64_t batchSize, std::vector<py::tuple>& src) {
  py::tuple dst(batchSize);
  const int64_t innerSize = src.size();
  for (int64_t i = 0; i < batchSize; ++i) {
    Sequence cur(innerSize);
    for (int64_t j = 0; j < innerSize; ++j) {
      cur[j] = std::move(src[j][i]);
    }
    dst[i] = std::move(cur);
  }
  return dst;
}

// Implementation of unstackFields.
// batchTuple preserves the tag for each tuple node which indicates whether the tuple nodes should be unstacked.
// tupleIndex is the index to track the current tuple node.
py::tuple unstackFieldsImpl(
    const py::handle& input, int64_t batchSize, int64_t dim, const std::vector<bool>& batchTuple, size_t& tupleIndex) {
  if (py::isinstance<py::tuple>(input)) {
    const py::tuple& src = py::reinterpret_borrow<py::tuple>(input);
    if (batchTuple[tupleIndex++]) {
      assert(src.size() == batchSize);
      return src;
    }
    const int64_t n = src.size();
    std::vector<py::tuple> children(n);
    for (int64_t i = 0; i < n; ++i) {
      children[i] = unstackFieldsImpl(src[i], batchSize, dim, batchTuple, tupleIndex);
    }
    return unstackSequence<py::tuple>(batchSize, children);
  }

  if (py::isinstance<py::list>(input)) {
    const py::list& src = py::reinterpret_borrow<py::list>(input);
    const int64_t n = src.size();
    std::vector<py::tuple> children(n);
    for (int64_t i = 0; i < n; ++i) {
      children[i] = unstackFieldsImpl(src[i], batchSize, dim, batchTuple, tupleIndex);
    }
    return unstackSequence<py::list>(batchSize, children);
  }

  if (py::isinstance<py::dict>(input)) {
    const py::dict& src = py::reinterpret_borrow<py::dict>(input);
    py::tuple dst(batchSize);
    for (int64_t i = 0; i < batchSize; ++i) {
      dst[i] = py::dict();
    }
    for (const auto& [k, v] : src) {
      py::list cur = unstackFieldsImpl(v, batchSize, dim, batchTuple, tupleIndex);
      assert(cur.size() == batchSize);
      for (int64_t i = 0; i < batchSize; ++i) {
        py::dict ret = py::reinterpret_borrow<py::dict>(dst[i]);
        ret[k] = std::move(cur[i]);
      }
    }
    return dst;
  }

  const auto src = rpc::tryFromPython(input);
  if (src) {
    py::tuple dst(batchSize);
    std::vector<rpc::Tensor> srcList = rpc::unbind(*src, dim);
    assert(srcList.size() == batchSize);
    for (int64_t i = 0; i < batchSize; ++i) {
      dst[i] = rpc::toPython(srcList[i]);
    }
    return dst;
  } else {
    return py::none();
  }
}

} // namespace

py::object squeezeFields(const py::handle& input, int64_t dim) {
  return squeezeFieldsImpl(input, dim).first;
}

py::object unsqueezeFields(const py::handle& input, int64_t dim) {
  return mapNested(
      [dim](const py::handle& input) -> py::object {
        const auto src = rpc::tryFromPython(input);
        return src ? rpc::toPython(src->unsqueeze(dim)) : py::make_tuple(input);
      },
      input);
}

py::object stackFields(const py::tuple& input, int64_t dim) {
  assert(!input.empty());
  if (input.size() == 1) {
    return unsqueezeFields(input[0], dim);
  }

  const int64_t batchSize = input.size();
  std::vector<std::vector<rpc::Tensor>> tensors;
  std::vector<py::tuple> objects;
  size_t tensorIndex = 0;
  size_t objectIndex = 0;

  // Flatten the input for stacking.
  for (int64_t i = 0; i < batchSize; ++i) {
    tensorIndex = 0;
    objectIndex = 0;
    visitNested(
        [batchSize, i, &tensors, &objects, &tensorIndex, &objectIndex](const py::handle& input) {
          const auto src = rpc::tryFromPython(input);
          if (src) {
            std::vector<rpc::Tensor>& cur =
                tensorIndex < tensors.size() ? tensors.at(tensorIndex) : tensors.emplace_back(batchSize);
            cur[i] = *src;
            ++tensorIndex;
          } else {
            py::tuple& cur = objectIndex < objects.size() ? objects.at(objectIndex) : objects.emplace_back(batchSize);
            cur[i] = input;
            ++objectIndex;
          }
        },
        input[i]);
  }

  std::vector<rpc::Tensor> stackedTensors;
  stackedTensors.reserve(tensors.size());
  for (const auto& cur : tensors) {
    const auto curSizes = cur[0].sizes();
    if (curSizes.size() > 0) {
      // torch::cat is much faster than torch::stack.
      // https://github.com/pytorch/pytorch/issues/22462
      std::vector<int64_t> newSizes(curSizes.size() + 1);
      newSizes[0] = batchSize;
      std::copy(curSizes.cbegin(), curSizes.cend(), newSizes.begin() + 1);
      stackedTensors.push_back(rpc::cat(cur, dim).view(rpc::IntArrayRef(newSizes.data(), newSizes.size())));
    } else {
      stackedTensors.push_back(rpc::stack(cur, dim));
    }
  }

  // Recover the nested structure as input[0].
  tensorIndex = 0;
  objectIndex = 0;
  return mapNested(
      [&stackedTensors, &objects, &tensorIndex, &objectIndex](const py::handle& input) -> py::object {
        const auto src = rpc::tryFromPython(input);
        py::object ret;
        if (src) {
          ret = rpc::toPython(stackedTensors[tensorIndex]);
          ++tensorIndex;
        } else {
          ret = std::move(objects[objectIndex]);
          ++objectIndex;
        }
        return ret;
      },
      input[0]);
}

py::tuple unstackFields(const py::handle& input, int64_t batchSize, int64_t dim) {
  if (batchSize == 1) {
    return py::make_tuple(squeezeFields(input, dim));
  }
  std::vector<bool> batchTuple;
  prepareForUnstack(input, batchTuple);
  size_t tupleIndex = 0;
  return unstackFieldsImpl(input, batchSize, dim, batchTuple, tupleIndex);
}

} // namespace utils
} // namespace moolib
