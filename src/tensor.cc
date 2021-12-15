/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "tensor.h"

#include <torch/torch.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace rpc {

// pytorch had an API change where non-const grad() was renamed to mutable_grad().
// we detect and use the one that works
template<typename T = torch::Tensor>
constexpr bool grad_is_mutable = std::is_same_v<T&, decltype(std::declval<T>().grad())>;
template<typename T = torch::Tensor>
auto mutable_grad(T& v) -> std::enable_if_t<grad_is_mutable<T>, T&> {
  return v.grad();
}
template<typename T = torch::Tensor>
auto mutable_grad(T& v) -> std::enable_if_t<!grad_is_mutable<T>, T&> {
  return v.mutable_grad();
}

#define t impl.as<torch::Tensor>()

Tensor::Tensor() {
  impl.emplace<torch::Tensor>();
}
Tensor::Tensor(std::nullptr_t) {}

Tensor::Tensor(const Tensor& n) {
  impl.emplace<torch::Tensor>(n.impl.as<torch::Tensor>());
}
Tensor::Tensor(Tensor&& n) {
  impl.emplace<torch::Tensor>(std::move(n.impl.as<torch::Tensor>()));
}
Tensor& Tensor::operator=(const Tensor& n) {
  t = n.impl.as<torch::Tensor>();
  return *this;
}
Tensor& Tensor::operator=(Tensor&& n) {
  t = std::move(n.impl.as<torch::Tensor>());
  return *this;
}

#define w0c(n, r)                                                                                                      \
  r Tensor::n() const {                                                                                                \
    return cast(t.n());                                                                                                \
  }
#define w0cv(n)                                                                                                        \
  Tensor& Tensor::n() const {                                                                                          \
    cast(t.n());                                                                                                       \
    return *this;                                                                                                      \
  }
#define w1c(n, r, t0)                                                                                                  \
  r Tensor::n(t0 a0) const {                                                                                           \
    return cast(t.n(cast(std::forward<t0>(a0))));                                                                      \
  }

#define w0(n, r)                                                                                                       \
  r Tensor::n() {                                                                                                      \
    return cast(t.n());                                                                                                \
  }
#define w0v(n)                                                                                                         \
  Tensor& Tensor::n() {                                                                                                \
    cast(t.n());                                                                                                       \
    return *this;                                                                                                      \
  }
#define w1(n, r, t0)                                                                                                   \
  r Tensor::n(t0 a0) {                                                                                                 \
    return cast(t.n(cast(std::forward<t0>(a0))));                                                                      \
  }
#define w1v(n, t0)                                                                                                     \
  Tensor& Tensor::n(t0 a0) {                                                                                           \
    cast(t.n(cast(std::forward<t0>(a0))));                                                                             \
    return *this;                                                                                                      \
  }
#define w2(n, r, t0, t1)                                                                                               \
  r Tensor::n(t0 a0, t1 a1) {                                                                                          \
    return cast(t.n(cast(std::forward<t0>(a0)), cast(std::forward<t1>(a1))));                                          \
  }
#define w2v(n, t0, t1)                                                                                                 \
  Tensor& Tensor::n(t0 a0, t1 a1) {                                                                                    \
    t.n(cast(std::forward<t0>(a0)), cast(std::forward<t1>(a1)));                                                       \
    return *this;                                                                                                      \
  }
#define w2c(n, r, t0, t1)                                                                                              \
  r Tensor::n(t0 a0, t1 a1) const {                                                                                    \
    return cast(t.n(cast(std::forward<t0>(a0)), cast(std::forward<t1>(a1))));                                          \
  }
#define w3c(n, r, t0, t1, t2)                                                                                          \
  r Tensor::n(t0 a0, t1 a1, t2 a2) const {                                                                             \
    return cast(t.n(cast(std::forward<t0>(a0)), cast(std::forward<t1>(a1)), cast(std::forward<t2>(a2))));              \
  }

template<typename T>
decltype(auto) cast(T&& v) {
  return std::forward<T>(v);
}

int cast(torch::ScalarType v) {
  return (int)v;
}

IntArrayRef cast(torch::IntArrayRef v) {
  return {v.data(), v.size()};
}
torch::IntArrayRef cast(IntArrayRef v) {
  return {v.data(), v.size()};
}

Device cast(torch::Device d) {
  switch (d.type()) {
  case torch::kCPU:
    return {DeviceType::Cpu, d.index()};
  case torch::kCUDA:
    return {DeviceType::Cuda, d.index()};
  default:
    return {DeviceType::Unknown, d.index()};
  }
}

torch::DeviceType cast(DeviceType v) {
  switch (v) {
  case kCPU:
    return torch::kCPU;
  case kCUDA:
    return torch::kCUDA;
  default:
    throw std::runtime_error("cast to unknown device type");
  }
}

torch::Device cast(Device v) {
  return torch::Device(cast(v.type), v.index);
}

Tensor cast(torch::Tensor v) {
  Tensor r(nullptr);
  r.impl.emplace<torch::Tensor>(std::move(v));
  return r;
}
const torch::Tensor& cast(const Tensor& v) {
  return v.impl.as<torch::Tensor>();
}
torch::Tensor& cast(Tensor& v) {
  return v.impl.as<torch::Tensor>();
}

Device::Device(std::string_view str) {
  *this = cast(torch::Device(std::string(str)));
}

Tensor torchTensorToTensor(const torch::Tensor& v) {
  return cast(v);
}

torch::Tensor tensorToTorchTensor(Tensor&& v) {
  return std::move(v.impl.as<torch::Tensor>());
}

w0(data_ptr, void*);
w0c(device, Device);
w0c(scalar_type, int);
w0c(sizes, IntArrayRef);
w0c(strides, IntArrayRef);
w0c(dim, int64_t);
w1c(size, int64_t, int64_t);
w0c(is_cuda, bool);
w0c(itemsize, int);
w0c(pin_memory, Tensor);
w0c(defined, bool);
w0c(cpu, Tensor);
w2v(copy_, const Tensor&, bool);
w0c(sum, Tensor);
Tensor Tensor::mutable_grad() {
  return cast(rpc::mutable_grad(t));
}
w0c(grad, Tensor);
void Tensor::set_grad(Tensor v) {
  rpc::mutable_grad(t) = v.t;
}
w0v(detach_);
w0v(zero_);
w1v(mul_, float);
w1v(add_, const Tensor&);
w3c(to, Tensor, Device, bool, bool);
w0c(requires_grad, bool);
w0c(numel, int64_t);
w2c(select, Tensor, int64_t, int64_t);
w3c(narrow, Tensor, int64_t, int64_t, int64_t);

w1c(view, Tensor, IntArrayRef);

w1c(squeeze, Tensor, int64_t);
w1v(squeeze_, int64_t);

w1c(unsqueeze, Tensor, int64_t);
w1v(unsqueeze_, int64_t);

Tensor& Tensor::operator+=(const Tensor& n) {
  t += n.t;
  return *this;
}
Tensor& Tensor::operator*=(const Tensor& n) {
  t *= n.t;
  return *this;
}

Tensor Tensor::operator[](size_t index) {
  return cast(t[index]);
}

#undef t

Tensor zeros_like(const Tensor& n) {
  return cast(torch::zeros_like(cast(n)));
}
Tensor zeros_like(const Tensor& n, Device d) {
  return cast(torch::zeros_like(cast(n), cast(d)));
}
Tensor empty(IntArrayRef sizes, int dtype, Device d) {
  return cast(torch::empty(cast(sizes), torch::TensorOptions().dtype(torch::ScalarType(dtype)).device(cast(d))));
}
Tensor from_blob(int dtype, IntArrayRef sizes, void* data) {
  return cast(torch::from_blob(data, cast(sizes), torch::TensorOptions(torch::ScalarType(dtype))));
}
Tensor& min_out(Tensor& out, const Tensor& self, const Tensor& other) {
  torch::min_out(cast(out), cast(self), cast(other));
  return out;
}
Tensor& max_out(Tensor& out, const Tensor& self, const Tensor& other) {
  torch::max_out(cast(out), cast(self), cast(other));
  return out;
}

Tensor cat(const std::vector<Tensor>& tensors, int64_t dim) {
  std::vector<torch::Tensor> src;
  src.reserve(tensors.size());
  for (const auto& x : tensors) {
    src.push_back(cast(x));
  }
  return cast(torch::cat(src, dim));
}

Tensor stack(const std::vector<Tensor>& tensors, int64_t dim) {
  std::vector<torch::Tensor> src;
  src.reserve(tensors.size());
  for (const auto& x : tensors) {
    src.push_back(cast(x));
  }
  return cast(torch::stack(src, dim));
}

std::vector<Tensor> unbind(const Tensor& input, int64_t dim) {
  std::vector<torch::Tensor> dst = torch::unbind(cast(input), dim);
  std::vector<Tensor> ret;
  ret.reserve(dst.size());
  for (const auto& x : dst) {
    ret.push_back(cast(x));
  }
  return ret;
}

struct AllocationType {
  torch::DataPtr data;
  size_t bytes;
};

Allocator::Allocator() {
  impl.emplace<AllocationType>();
}
Allocator::Allocator(Device device, size_t bytes) {
  AllocationType& ac = impl.emplace<AllocationType>();
  if (device.type == kCPU) {
    ac.data = at::getCPUAllocator()->allocate(bytes);
    ac.bytes = bytes;
  } else if (device.type == kCUDA) {
#ifdef USE_CUDA
    ac.data = c10::cuda::CUDACachingAllocator::get()->allocate(bytes);
#else
    throw std::runtime_error("Cuda support is disabled; can not allocate cuda tensor");
#endif
    ac.bytes = bytes;
  } else {
    throw std::runtime_error("allocated on unknown device type");
  }
}
Allocator::Allocator(Allocator&& n) {
  impl.emplace<AllocationType>(std::move(n.impl.as<AllocationType>()));
}

std::byte* Allocator::data() const {
  return (std::byte*)impl.as<AllocationType>().data.get();
}
size_t Allocator::size() const {
  return impl.as<AllocationType>().bytes;
}

Tensor Allocator::set(int dtype, IntArrayRef sizes, IntArrayRef strides) {
  auto& ac = impl.as<AllocationType>();

  auto device = ac.data.device();
  torch::Storage storage(torch::Storage::use_byte_size_t(), ac.bytes, std::move(ac.data), nullptr, false);
  torch::Tensor t = torch::empty({0}, torch::TensorOptions(torch::ScalarType(dtype)).device(device))
                        .set_(std::move(storage), 0, cast(sizes), cast(strides));
  return cast(std::move(t));
}

AutoGradMode::AutoGradMode(bool enabled) {
  impl.emplace<torch::AutoGradMode>(enabled);
}
AutoGradMode::~AutoGradMode() {}

#ifdef USE_CUDA

bool CudaSupported() {
  return true;
}

CUDAStream::CUDAStream(std::nullptr_t) {}
CUDAStream::CUDAStream(const CUDAStream& n) {
  impl.emplace<c10::cuda::CUDAStream>(n.impl.as<c10::cuda::CUDAStream>());
}
CUDAStream::~CUDAStream() {}
void CUDAStream::synchronize() {
  impl.as<c10::cuda::CUDAStream>().synchronize();
}

CUDAStreamGuard::CUDAStreamGuard(const CUDAStream& n) {
  impl.emplace<c10::cuda::CUDAStreamGuard>(n.impl.as<c10::cuda::CUDAStream>());
}
CUDAStreamGuard::~CUDAStreamGuard() {}

CUDAStream getCurrentCUDAStream(int device_index) {
  CUDAStream r(nullptr);
  r.impl.emplace<c10::cuda::CUDAStream>(c10::cuda::getCurrentCUDAStream(device_index));
  return r;
}

#else

bool CudaSupported() {
  return false;
}

CUDAStream::CUDAStream(std::nullptr_t) {}
CUDAStream::CUDAStream(const CUDAStream& n) {}
CUDAStream::~CUDAStream() {}

CUDAStreamGuard::CUDAStreamGuard(const CUDAStream& n) {}
CUDAStreamGuard::~CUDAStreamGuard() {}

CUDAStream getCurrentCUDAStream(int device_index) {
  return {nullptr};
}

void CUDAStream::synchronize() {}

#endif

} // namespace rpc
