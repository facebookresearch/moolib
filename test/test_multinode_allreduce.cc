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

#include <thread>

template<torch::DeviceType device>
struct AllReduce {
  rpc::Rpc rpc;

  size_t myRank = 0;
  size_t worldSize = 0;

  torch::Tensor localData;
  std::atomic<size_t> chunksDone = 0;
  std::atomic<size_t> syncCount = 0;
  std::atomic_int calls = 0;
  void reduce(size_t sourcePeer, size_t offset, torch::Tensor data) {
    ++calls;
    size_t nextPeer = myRank == worldSize - 1 ? 0 : myRank + 1;
    auto ldata = localData.narrow(0, offset, data.size(0));
    ldata += data;
    if (sourcePeer == nextPeer) {
      for (size_t i = 0; i != worldSize; ++i) {
        if (i != myRank) {
          rpc.async(std::to_string(i), "share", offset, ldata);
        }
      }
      ++chunksDone;
      // fmt::printf("reduce: I have all data from %d for %d + %d (chunksDone is now %d)\n", sourcePeer, offset,
      // data.size(0), chunksDone.load());
    } else {
      rpc.async(std::to_string(nextPeer), "reduce", sourcePeer, offset, ldata);
    }
  }
  void share(size_t offset, torch::Tensor data) {
    ++calls;
    localData.narrow(0, offset, data.size(0)) = data;
    ++chunksDone;
    // fmt::printf("share: I got share data for %d + %d (chunksDone is now %d)\n", offset, data.size(0),
    // chunksDone.load());
  }

  void synchronize() {
    static std::atomic_int32_t counter = 0;
    for (size_t i = 0; i != worldSize; ++i) {
      if (i != myRank) {
        rpc.async(std::to_string(i), "sync", std::string(rpc.getName()) + "-" + std::to_string(++counter));
      }
    }
    while (syncCount < worldSize - 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    syncCount -= worldSize - 1;
    fmt::printf("Synchronized!\n");
  }

  AllReduce(size_t worldSize, size_t rank, std::string masterAddr) : myRank(rank), worldSize(worldSize) {

    rpc.define<void(size_t, size_t, torch::Tensor)>(
        "reduce",
        [this](size_t sourcePeer, size_t offset, torch::Tensor data) { reduce(sourcePeer, offset, std::move(data)); });
    rpc.define<void(size_t, torch::Tensor)>(
        "share", [this](size_t offset, torch::Tensor data) { share(offset, std::move(data)); });
    rpc.define<void(std::string)>("sync", [this](std::string id) { ++syncCount; });

    rpc.setName(std::to_string(rank));

    if (rank == 0) {
      rpc.listen(masterAddr);
    } else {
      rpc.connect(masterAddr);
    }

    for (int i = 0; i < 20; i += 1) {

      chunksDone = 0;

      size_t dataSize = 400 + 1024 * 128 * i;

      localData = torch::randn({(int64_t)dataSize}, device);

      synchronize();

      Timer tx;

      size_t remainingData = dataSize;
      size_t offset = 0;
      for (size_t i = 0; i != worldSize; ++i) {
        size_t div = (worldSize - i);
        size_t chunkSize = (remainingData + div - 1) / div;
        remainingData -= chunkSize;

        if (i == myRank) {
          size_t nextPeer = i == worldSize - 1 ? 0 : i + 1;
          rpc.async(std::to_string(nextPeer), "reduce", i, offset, localData.narrow(0, offset, chunkSize));
        }

        offset += chunkSize;
      }

      ASSERT(remainingData == 0);

      while (chunksDone != worldSize) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      fmt::printf("Local AllReduce done\n");

      float time = tx.elapsed();

      //      for (int i = 0; i != 10; ++i) {
      //        ASSERT(chunksDone == worldSize);
      //        ASSERT(calls == (worldSize - 1) * 2);
      //        std::this_thread::sleep_for(std::chrono::milliseconds(i));
      //      }
      synchronize();

      //      for (size_t i = 0; i != dataSize; ++i) {
      //        uint64_t sum = 0;
      //        for (auto& v : originalLocalData) {
      //          sum += v[i];
      //        }
      //        for (auto& v : finalData) {
      //          ASSERT(sum == v[i]);
      //        }
      //      }

      ASSERT(chunksDone == worldSize);

      fmt::printf("AllReduce %d done!  Sum %g\n", i, localData.sum().template item<float>());

      int thiscalls = calls;
      calls = 0;

      fmt::printf("AllReduce %gs, %gM/s, calls: %d\n", time, (dataSize / time) / 1024 / 1024, thiscalls);
    }

    rpc.debugInfo();
  }
};

using AllReduceCpu = AllReduce<torch::kCPU>;
using AllReduceCuda = AllReduce<torch::kCUDA>;

#include <signal.h>

int main() {
  struct sigaction act;
  act.sa_handler = SIG_IGN;
  sigaction(SIGPIPE, &act, NULL);

  auto env = [&](const char* name) {
    const char* value = std::getenv(name);
    if (!value) {
      fmt::printf("Required env var %s not set\n", name);
      std::exit(-1);
    }
    return value;
  };

  int worldSize = std::atoi(env("WORLD_SIZE"));
  int rank = std::atoi(env("RANK"));
  std::string masterAddr = std::string(env("MASTER_ADDR")) + ":" + std::string(env("MASTER_PORT"));

  fmt::printf("World size: %d\nRank: %d\nMaster address: %s\n", worldSize, rank, masterAddr);
  fflush(stdout);

  RUNARG(AllReduceCpu, worldSize, rank, masterAddr);
  // RUNARG(AllReduceCuda, worldSize, rank, masterAddr);

  quit();
  return 0;
}
