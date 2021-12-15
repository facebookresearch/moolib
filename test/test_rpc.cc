/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test.h"

#include "fmt/printf.h"
#include "pytorch.h"
#include "rpc.h"

#include <cstdint>
#include <deque>
#include <thread>

std::string localAddr = "127.0.0.1:8888";
// std::string localAddr = "shm://rpctest";

struct RpcTest {
  RpcTest(int nPeers) {
    peers.resize(nPeers);
  }
  std::deque<rpc::Rpc> peers;
};

struct Hello : RpcTest {
  std::string msg_;
  Hello() : RpcTest(2) {
    peers[0].define<std::string(std::string)>("hello", [this](std::string msg) {
      msg_ = msg;
      return msg + " back at ya";
    });
    peers[0].setName("host");
    peers[0].listen(localAddr);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    peers[1].connect(localAddr);

    std::string retval = peers[1].sync<std::string>("host", "hello", std::string("hello world"));

    ASSERT(msg_ == "hello world");
    ASSERT(retval == "hello world back at ya");
  }
};

struct Sum : RpcTest {
  std::atomic_uint64_t sum_ = 0;
  Sum() : RpcTest(2) {
    peers[0].define<void(uint64_t)>("add", [this](uint64_t x) { sum_ += x; });
    peers[0].setName("host");
    peers[0].listen(localAddr);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    uint64_t localSum = 0;
    for (size_t j = 1; j != peers.size(); ++j) {
      peers[j].connect(localAddr);

      for (uint64_t i = 0; i != 1000; ++i) {
        localSum += i;
        peers[j].sync("host", "add", i);
      }
    }

    ASSERT(sum_ == localSum);
  }
};

struct AsyncSum : RpcTest {
  std::atomic<uint64_t> sum_ = 0;
  AsyncSum() : RpcTest(8) {
    peers[0].define<void(uint64_t)>("add", [this](uint64_t x) { sum_ += x; });
    peers[0].setName("host");
    peers[0].listen(localAddr);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for (size_t i = 1; i != peers.size(); ++i) {
      peers[i].connect(localAddr);
    }

    std::vector<std::future<void>> futures;

    uint64_t localSum = 0;
    size_t futIndex = 0;
    for (int iteration = 0; iteration != 40; ++iteration) {
      for (size_t ic = 1; ic != peers.size(); ++ic) {
        for (uint64_t i = 0; i != 100; ++i) {
          localSum += i << ic;
          futures.push_back(peers[ic].async("host", "add", i << ic));
        }
      }
      for (size_t i = futIndex; i != futures.size() / 2; ++i) {
        futures[i].wait();
      }
      futIndex = futures.size() / 2;
    }
    for (size_t i = futIndex; i != futures.size(); ++i) {
      futures[i].wait();
    }

    fmt::printf("sum is %#x, localsum is %#x\n", sum_.load(), localSum);

    ASSERT(sum_ == localSum);
  }
};

struct AllReduce : RpcTest {
  std::vector<std::vector<uint64_t>> localData;
  std::vector<std::vector<uint64_t>> finalData;
  std::atomic<size_t> chunksDone = 0;
  std::atomic_int calls = 0;
  void reduce(size_t peerIndex, size_t sourcePeer, size_t offset, std::basic_string_view<uint64_t> data) {
    ++calls;
    size_t nextPeer = peerIndex == peers.size() - 1 ? 0 : peerIndex + 1;
    auto& ldata = localData.at(peerIndex);
    for (size_t i = 0; i != data.size(); ++i) {
      ldata[offset + i] += data[i];
    }
    data = std::basic_string_view<uint64_t>(ldata.data() + offset, data.size());
    if (sourcePeer == nextPeer) {
      // fmt::printf("peer %d has all data from %d for %d + %d\n", peerIndex, sourcePeer, offset, data.size());
      for (size_t i = 0; i != peers.size(); ++i) {
        if (i != peerIndex) {
          peers[peerIndex].async(peers[i].getName(), "share", offset, data);
        }
      }
      auto& fdata = finalData.at(peerIndex);
      for (size_t i = 0; i != data.size(); ++i) {
        fdata[offset + i] = data[i];
      }
      ++chunksDone;
    } else {
      peers[peerIndex].async(peers[nextPeer].getName(), "reduce", sourcePeer, offset, data);
    }
  }
  void share(size_t peerIndex, size_t offset, std::basic_string_view<uint64_t> data) {
    ++calls;
    // fmt::printf("peer %d got share data for %d + %d\n", peerIndex, offset, data.size());
    auto& fdata = finalData.at(peerIndex);
    for (size_t i = 0; i != data.size(); ++i) {
      fdata[offset + i] = data[i];
    }
    ++chunksDone;
  }
  AllReduce() : RpcTest(20) {

    for (size_t i = 0; i != peers.size(); ++i) {
      peers[i].define<void(size_t, size_t, std::basic_string_view<uint64_t>)>(
          "reduce", [this, i](size_t sourcePeer, size_t offset, std::basic_string_view<uint64_t> data) {
            reduce(i, sourcePeer, offset, std::move(data));
          });
      peers[i].define<void(size_t, std::basic_string_view<uint64_t>)>(
          "share",
          [this, i](size_t offset, std::basic_string_view<uint64_t> data) { share(i, offset, std::move(data)); });
    }

    peers[0].setName("host");
    peers[0].listen(localAddr);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for (size_t i = 1; i != peers.size(); ++i) {
      peers[i].setName("peer " + std::to_string(i));
      peers[i].connect(localAddr);
    }

    uint64_t x = 42;

    for (int i = 0; i < 10; i += 3) {

      chunksDone = 0;
      localData.clear();
      finalData.clear();
      localData.resize(peers.size());
      finalData.resize(peers.size());

      size_t dataSize = 40 + 1024 * 128 * i;

      for (size_t i = 0; i != peers.size(); ++i) {
        localData[i].resize(dataSize);
        finalData[i].resize(localData[i].size());
        for (size_t j = 0; j != localData[i].size(); ++j) {
          localData[i][j] = (i + 1) * (j + 1) + x;
          // x = x * 48271 + 1;
          x += 48271;
        }
      }

      auto originalLocalData = localData;

      Timer tx;

      size_t remainingData = dataSize;
      size_t offset = 0;
      for (size_t i = 0; i != peers.size(); ++i) {
        size_t div = (peers.size() - i);
        size_t chunkSize = (remainingData + div - 1) / div;
        remainingData -= chunkSize;

        size_t nextPeer = i == peers.size() - 1 ? 0 : i + 1;
        peers[i].async(
            peers[nextPeer].getName(), "reduce", i, offset,
            std::basic_string_view<uint64_t>(localData[i].data() + offset, chunkSize));

        offset += chunkSize;
      }

      ASSERT(remainingData == 0);

      while (chunksDone != peers.size() * peers.size()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }

      float time = tx.elapsed();

      for (int i = 0; i != 10; ++i) {
        ASSERT(chunksDone == peers.size() * peers.size());
        ASSERT(calls == peers.size() * (peers.size() - 1) * 2);
        std::this_thread::sleep_for(std::chrono::milliseconds(i));
      }

      for (size_t i = 0; i != dataSize; ++i) {
        uint64_t sum = 0;
        for (auto& v : originalLocalData) {
          sum += v[i];
        }
        for (auto& v : finalData) {
          ASSERT(sum == v[i]);
        }
      }

      ASSERT(chunksDone == peers.size() * peers.size());

      fmt::printf("AllReduce %d done!\n", i);

      int thiscalls = calls;
      calls = 0;

      fmt::printf("AllReduce %gs, %gM/s, calls: %d\n", time, (dataSize / time) / 1024 / 1024, thiscalls);
    }
  }
};

template<bool Cuda>
struct Tensor : RpcTest {
  std::mutex mut;
  torch::Tensor sum = torch::zeros({1, 2, 3});
  torch::Tensor mulsum = torch::zeros({1, 2, 3});
  Tensor() : RpcTest(2) {
    if (Cuda) {
      sum = sum.cuda();
      mulsum = mulsum.cuda();
    }
    auto device = Cuda ? torch::kCUDA : torch::kCPU;
    torch::AutoGradMode ag(false);
    peers[0].template define<void(std::unordered_map<std::string_view, torch::Tensor>)>(
        "op", [this](std::unordered_map<std::string_view, torch::Tensor> x) {
          std::lock_guard l(mut);
          sum += x["add"];
          sum -= x["sub"];
          mulsum += x["mul"];
        });
    auto hostName = peers[0].getName();
    peers[0].listen(localAddr);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for (size_t i = 1; i != peers.size(); ++i) {
      peers[i].connect(localAddr);
    }

    std::vector<std::future<void>> futures;

    torch::Tensor localSum = torch::zeros({1, 2, 3}, device);
    torch::Tensor localMulSum = torch::zeros({1, 2, 3}, device);
    size_t futIndex = 0;
    for (int iteration = 0; iteration != 1; ++iteration) {
      for (size_t ic = 1; ic != peers.size(); ++ic) {
        for (uint64_t i = 0; i != 1; ++i) {
          torch::Tensor add = torch::randn({1, 2, 3}, device);
          torch::Tensor sub = torch::randn({1, 2, 3}, device);
          torch::Tensor mul = torch::randn({1, 2, 3}, device);
          localSum += add;
          localSum -= sub;
          localMulSum += mul;
          std::unordered_map<std::string, torch::Tensor> map;
          map["add"] = add;
          map["sub"] = sub;
          map["mul"] = mul;
          futures.push_back(peers[ic].async(hostName, "op", map));
        }
      }
      for (size_t i = futIndex; i != futures.size() / 2; ++i) {
        futures[i].wait();
      }
      futIndex = futures.size() / 2;
    }
    for (size_t i = futIndex; i != futures.size(); ++i) {
      futures[i].wait();
    }

    localSum *= localMulSum;
    sum *= mulsum;

    fmt::printf("sum is %g, localsum is %g\n", sum.sum().template item<float>(), localSum.sum().item<float>());

    ASSERT((sum.sub(localSum).abs() >= 1e-2).sum().template item<long>() == 0);
  }
};

using CpuTensor = Tensor<false>;
using CudaTensor = Tensor<true>;

#include <signal.h>

int main() {
  struct sigaction act;
  act.sa_handler = SIG_IGN;
  sigaction(SIGPIPE, &act, NULL);

  RUN(Hello);
  RUN(Sum);
  RUN(AsyncSum);
  RUN(AllReduce);

  RUN(CpuTensor);
  // RUN(CudaTensor);

  quit();
  return 0;
}
