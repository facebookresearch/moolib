
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <optional>
#include <random>

#include <fmt/printf.h>

namespace batchsizefinder {

struct Timer {
  std::chrono::steady_clock::time_point start;
  Timer() {
    reset();
  }
  void reset() {
    start = std::chrono::steady_clock::now();
  }
  float elapsedAt(std::chrono::steady_clock::time_point now) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1>>>(now - start).count();
  }
  float elapsed() {
    return elapsedAt(std::chrono::steady_clock::now());
  }
  float elapsedReset() {
    auto now = std::chrono::steady_clock::now();
    float r = elapsedAt(now);
    start = now;
    return r;
  }
};

float defaultScore(float latency, int bs) {
  return latency / 400 - std::log(bs / latency);
}

template<typename Prepare, typename Forward, typename Score>
int find(
    torch::Device device, Prepare&& prepare, Forward&& forward, int minBatchSize, int maxBatchsize, float maxTimeMs,
    Score&& scoreFunction) {
  torch::NoGradGuard ng;
  bool isCuda = device.is_cuda();
  std::optional<c10::cuda::CUDAStreamGuard> g;
  if (isCuda) {
    g.emplace(c10::cuda::getStreamFromPool(false, device.index()));
  } else {
    // throw std::runtime_error("findBatchSize on non-cuda device is not meaningful");
  }
  auto input = prepare(1);
  auto call = [&]() {
    forward(input);
    if (isCuda) {
      g->current_stream().synchronize();
    }
  };
  fmt::printf("Finding batch size\n");
  // warm up
  for (int i = 0; i != 10; ++i) {
    call();
  }
  Timer t;
  for (int i = 0; i != 10; ++i) {
    call();
  }
  float call1 = t.elapsed() / 10.0f * 1000.0f;
  fmt::printf("Base latency: %gms\n", call1);

  float maxms = maxTimeMs;
  int maxbs = maxBatchsize;

  struct I {
    float latency = 0.0f;
    int size = 0;
    int n = 0;
    bool isBad = false;
  };

  auto scorex = [&](auto& x) { return scoreFunction(x.latency / x.n, x.size); };

  std::map<int, I> li;

  int best = 0;
  float bestScore = std::numeric_limits<float>::infinity();

  auto eval = [&](int i) {
    input = prepare(i);
    int badcount = 0;
    float latency = 0.0f;
    int n = 2;
    for (int j = 0; j != n; ++j) {
      call();
    }
    for (int j = 0; j != n; ++j) {
      t.reset();
      call();
      float ms = t.elapsed() * 1000;
      latency += ms;
      if (ms > maxms || i > maxbs || i < minBatchSize) {
        ++badcount;
      }
    }
    auto& x = li[i];
    x.size = i;
    x.latency += latency;
    x.n += n;
    x.isBad = badcount >= n;
    float score = scorex(x);
    if (!x.isBad && score < bestScore) {
      bestScore = score;
      best = i;
    }
    return badcount < n;
  };

  for (int i = std::max(minBatchSize, 1);; i += (i + 3) / 4) {
    if (!eval(i)) {
      break;
    }
  }
  std::minstd_rand rng(std::random_device{}());

  auto expandNear = [&](int k) {
    int r = 0;
    auto i = li.find(k);
    if (i != li.end()) {
      auto search = [&](auto begin, auto end) {
        int b = begin->first;
        int e;
        if (end == li.end()) {
          e = std::prev(end)->first;
        } else {
          e = end->first;
        }
        b = std::max(b, i->first - 3);
        e = std::max(b, i->first + 6);
        for (int i = b; i != e; ++i) {
          if (li.find(i) != li.end()) {
            continue;
          }
          ++r;
          if (!eval(i)) {
            break;
          }
        }
      };
      search(i, std::next(i));
      if (i != li.begin()) {
        search(std::prev(i), i);
      }
    }
    return r;
  };

  for (int j = 0; j != 4; ++j) {
    int expands = 12;
    for (int k = 0; k != 12; ++k) {
      float sum = 0.0f;
      std::vector<std::tuple<float, int, int>> list;
      float minweight = std::numeric_limits<float>::infinity();
      for (auto& [k, v] : li) {
        if (!v.isBad) {
          minweight = std::min(minweight, scorex(v));
        }
      }
      for (auto i = li.begin();;) {
        auto next = std::next(i);
        if (next == li.end()) {
          break;
        }
        if (i->second.isBad && next->second.isBad) {
          i = next;
          continue;
        }
        int from = i->first + 1;
        int to = next->first;
        if (to - from > 0) {
          float weight = std::min(scorex(i->second), scorex(next->second)) - minweight;
          weight = 1.0f / std::min(std::exp(weight * 4), 1e9f);
          weight *= to - from;
          list.emplace_back(weight, from, to);
          sum += weight;
        }
        i = next;
      }
      if (list.size() > 0 && sum > 0.0f) {
        float val = std::uniform_real_distribution<float>(0.0f, sum)(rng);
        for (auto& [weight, from, to] : list) {
          val -= weight;
          if (val <= 0) {
            int k = std::uniform_int_distribution<int>(from, to - 1)(rng);
            eval(k);
            if (expands > 0) {
              expands -= expandNear(k);
            }
            break;
          }
        }
      }
    }
    if (best) {
      expandNear(best);
    }
    std::vector<std::tuple<float, int>> sorted;
    for (auto& [k, v] : li) {
      if (!v.isBad) {
        sorted.emplace_back(scorex(v), k);
      }
    }
    std::sort(sorted.begin(), sorted.end());
    for (size_t i = 0; i != sorted.size() && i < 10; ++i) {
      int k = std::get<1>(sorted[i]);
      if (li[k].n < 8) {
        eval(k);
      }
    }
  }

  for (auto& [k, v] : li) {
    fmt::printf(
        "Batch size %d, evals %d latency %fms throughput %g score %g\n", k, v.n, v.latency / v.n,
        v.size / (v.latency / v.n), scorex(v));
  }

  fmt::printf(
      "Found best batch size of %d with evals %d latency %fms "
      "throughput %g score %g\n",
      best, li[best].n, li[best].latency / li[best].n, li[best].size / (li[best].latency / li[best].n),
      scorex(li[best]));
  return best;
}

template<typename Prepare, typename Forward>
int find(
    torch::Device device, Prepare&& prepare, Forward&& forward, int minBatchSize, int maxBatchsize, float maxTimeMs) {
  return find(
      device, std::forward<Prepare>(prepare), std::forward<Forward>(forward), minBatchSize, maxBatchsize, maxTimeMs,
      defaultScore);
}

} // namespace batchsizefinder
