/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "any.h"

#include <optional>
#include <string_view>
#include <vector>

namespace rpc {

template<typename T>
using ArrayRef = std::basic_string_view<T>;

using IntArrayRef = ArrayRef<int64_t>;

enum class DeviceType { Cpu, Cuda, Unknown };

struct Device {
  DeviceType type = DeviceType::Unknown;
  int index = -1;
  Device() = default;
  Device(DeviceType type) : type(type) {}
  Device(DeviceType type, int index) : type(type), index(index) {}
  Device(std::string_view str);
};

struct Tensor {
  moolib::Any<32> impl;

  Tensor();
  Tensor(std::nullptr_t);

  Tensor(const Tensor&);
  Tensor(Tensor&&);
  Tensor& operator=(const Tensor&);
  Tensor& operator=(Tensor&&);

  Device device() const;
  int scalar_type() const;
  IntArrayRef sizes() const;
  IntArrayRef strides() const;
  int64_t dim() const;
  int64_t size(int64_t dim) const;

  bool is_cuda() const;
  int itemsize() const;

  void* data_ptr();

  template<typename T>
  T* data() {
    return (T*)data_ptr();
  }
  template<typename T>
  T item() {
    return *data<T>();
  }

  // These constant values match with pytorch
  static constexpr int kUInt8 = 0;
  static constexpr int kInt8 = 1;
  static constexpr int kInt16 = 2;
  static constexpr int kInt32 = 3;
  static constexpr int kInt64 = 4;
  static constexpr int kFloat16 = 5;
  static constexpr int kFloat32 = 6;
  static constexpr int kFloat64 = 7;
  static constexpr int kBool = 11;

  Tensor pin_memory() const;
  bool defined() const;
  Tensor cpu() const;
  Tensor& copy_(const Tensor& n, bool non_blocking = false);
  Tensor sum() const;

  Tensor& operator+=(const Tensor&);
  Tensor& operator*=(const Tensor&);

  Tensor mutable_grad();
  Tensor grad() const;
  void set_grad(Tensor);
  Tensor& detach_();
  Tensor& zero_();
  Tensor& mul_(float n);
  Tensor& add_(const Tensor&);
  int64_t numel() const;
  Tensor select(int64_t dim, int64_t index) const;
  Tensor narrow(int64_t dim, int64_t start, int64_t length) const;

  Tensor view(IntArrayRef sizes) const;

  Tensor squeeze(int64_t dim) const;
  Tensor& squeeze_(int64_t dim);

  Tensor unsqueeze(int64_t dim) const;
  Tensor& unsqueeze_(int64_t dim);

  Tensor to(Device device, bool non_blocking = false, bool copy = false) const;

  bool requires_grad() const;

  Tensor operator[](size_t index);
};

struct Allocator {
  moolib::Any<40> impl;
  Allocator();
  Allocator(Device device, size_t bytes);
  Allocator(Allocator&&);
  std::byte* data() const;
  size_t size() const;
  Tensor set(int dtype, IntArrayRef sizes, IntArrayRef strides);
};

constexpr auto kCPU = DeviceType::Cpu;
constexpr auto kCUDA = DeviceType::Cuda;

Tensor zeros_like(const Tensor&);
Tensor zeros_like(const Tensor&, Device);
Tensor empty(IntArrayRef sizes, int dtype, Device d);

Tensor from_blob(int dtype, IntArrayRef sizes, void* data);
Tensor& min_out(Tensor& out, const Tensor&, const Tensor&);
Tensor& max_out(Tensor& out, const Tensor&, const Tensor&);

Tensor cat(const std::vector<Tensor>& tensors, int64_t dim);
Tensor stack(const std::vector<Tensor>& tensors, int64_t dim);
std::vector<Tensor> unbind(const Tensor& input, int64_t dim);

struct AutoGradMode {
  moolib::Any<8> impl;
  AutoGradMode(bool enabled);
  ~AutoGradMode();
};

template<typename X>
void serialize(X& x, const Tensor& v) {
  x.addTensor(v, x.tell());
  x(v.scalar_type(), v.sizes(), v.strides());
}

template<typename X>
void serialize(X& x, Tensor& v) {
  decltype(v.scalar_type()) dtype;
  decltype(v.sizes()) sizes;
  decltype(v.strides()) strides;
  x(dtype, sizes, strides);
  v = std::move(x.getTensor().tensor);
}

bool CudaSupported();

struct CUDAStream {
  moolib::Any<16> impl;
  CUDAStream(std::nullptr_t);
  CUDAStream(const CUDAStream&);
  ~CUDAStream();
  void synchronize();
};

struct CUDAStreamGuard {
  moolib::Any<16> impl;
  CUDAStreamGuard(const CUDAStream&);
  ~CUDAStreamGuard();
};

CUDAStream getCurrentCUDAStream(int device_index = -1);

} // namespace rpc
