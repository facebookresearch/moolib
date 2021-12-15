/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "util.h"

#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace moolib {

struct BrokerService {

  struct Peer {
    std::string name;
    std::chrono::steady_clock::time_point lastPing;
    std::chrono::steady_clock::duration timeoutDuration;
    std::optional<Future<std::pair<uint32_t, int32_t>>> syncFuture;
    std::optional<Future<void>> updateFuture;
    int32_t sortOrder = 0;
    bool active = false;
    size_t creationOrder = 0;
  };

  struct Group {
    std::mutex mutex;
    std::string name;
    std::unordered_map<std::string, Peer> peers;
    bool needsUpdate = false;
    std::chrono::steady_clock::time_point lastUpdate;
    uint32_t syncId = 1;
    size_t updateCount = 0;
    size_t orderCounter = 0;

    std::vector<std::string> active;

    Peer& getPeer(std::string name) {
      auto i = peers.try_emplace(name);
      if (i.second) {
        auto& p = i.first->second;
        p.name = name;
        p.creationOrder = orderCounter++;
      }
      return i.first->second;
    }
  };

  std::mutex groupsMutex;
  std::unordered_map<std::string, Group> groups;

  Group& getGroup(const std::string& name) {
    std::lock_guard l(groupsMutex);
    auto i = groups.try_emplace(name);
    if (i.second) {
      auto& g = i.first->second;
      g.name = name;
    }
    return i.first->second;
  }

  std::unordered_set<Group*> syncSet;
  std::chrono::steady_clock::time_point lastCheckTimeouts;

  uint32_t nextSyncId = random<uint32_t>();

  std::vector<Group*> tmpGroups;
  std::vector<Peer*> tmpPeers;

  rpc::Rpc* rpc = nullptr;

  BrokerService(rpc::Rpc& rpc) : rpc(&rpc) {
    setup();
  }
  ~BrokerService() {}

  template<typename T, typename... Args>
  Future<T> call(std::string_view peerName, std::string_view funcName, Args&&... args) {
    return callImpl<T>(*rpc, peerName, funcName, std::forward<Args>(args)...);
  }

  void setup() {

    rpc->define<size_t(std::string)>("BrokerService::groupSize", [this](std::string group) {
      log.info("groupSize called!\n");
      auto& g = getGroup(group);
      std::lock_guard l(g.mutex);
      return g.peers.size();
    });

    rpc->define<uint32_t(std::string, std::string, uint32_t)>(
        "BrokerService::ping", [this](std::string group, std::string name, uint32_t timeoutMilliseconds) {
          auto& g = getGroup(group);
          std::lock_guard l(g.mutex);
          auto& p = g.getPeer(name);
          p.lastPing = std::chrono::steady_clock::now();
          p.timeoutDuration = std::chrono::milliseconds(timeoutMilliseconds);
          if (!p.active) {
            g.needsUpdate = true;
          }
          // log("got ping for %s::%s\n", group, name);
          return g.syncId;
        });

    rpc->define<void(std::string)>("BrokerService::resync", [this](std::string group) {
      auto& g = getGroup(group);
      std::lock_guard l(g.mutex);
      if (!g.needsUpdate) {
        log.info("Got resync request for %s\n", group);
        g.needsUpdate = true;
      }
    });
  }

  void update() {

    auto now = std::chrono::steady_clock::now();

    if (!syncSet.empty()) {

      for (auto i = syncSet.begin(); i != syncSet.end();) {
        auto& g = **i;
        std::lock_guard l(g.mutex);
        size_t total = 0;
        size_t ready = 0;
        for ([[maybe_unused]] auto& [pname, p] : g.peers) {
          if (p.syncFuture) {
            ++total;
            if (*p.syncFuture) {
              if ((*p.syncFuture)->first == g.syncId) {
                ++ready;
              } else {
                log.info("bad sync id?? got %#x expected %#x", (*p.syncFuture)->first, g.syncId);
                --total;
              }
            }
          }
        }
        // log("Sync midway %s %d/%d in %gs\n", g.name, ready, total, seconds(now - g.lastUpdate));
        if (ready >= total || now - g.lastUpdate >= std::chrono::seconds(1)) {
          log.info("Sync %s %d/%d in %gs\n", g.name, ready, total, seconds(now - g.lastUpdate));

          tmpPeers.clear();
          for ([[maybe_unused]] auto& [pname, p] : g.peers) {
            if (p.syncFuture && *p.syncFuture && (*p.syncFuture)->first == g.syncId) {
              p.sortOrder = (*p.syncFuture)->second;
              tmpPeers.push_back(&p);
              p.active = true;
            } else {
              p.active = false;
            }
          }
          std::sort(tmpPeers.begin(), tmpPeers.end(), [](Peer* a, Peer* b) {
            if (a->sortOrder == b->sortOrder) {
              return a->creationOrder < b->creationOrder;
            }
            return a->sortOrder < b->sortOrder;
          });
          g.active.clear();
          for (auto* p : tmpPeers) {
            log.info("%s with sort order %d\n", p->name, p->sortOrder);
            g.active.push_back(p->name);
          }
          if (!g.active.empty()) {
            log.info("%s is the master\n", g.active.front());
          }
          for (auto* p : tmpPeers) {
            p->updateFuture = call<void>(p->name, "GroupService::update", g.name, g.syncId, g.active);
          }

          i = syncSet.erase(i);
        } else {
          ++i;
        }
      }
    }

    if (now - lastCheckTimeouts < std::chrono::milliseconds(500)) {
      return;
    }

    lastCheckTimeouts = now;
    tmpGroups.clear();
    {
      std::lock_guard l(groupsMutex);
      for (auto& [gname, g] : groups) {
        tmpGroups.push_back(&g);
      }
    }
    for (auto* pg : tmpGroups) {
      auto& g = *pg;
      std::lock_guard l2(g.mutex);
      for (auto i = g.peers.begin(); i != g.peers.end();) {
        auto& p = i->second;
        if (now - p.lastPing >= p.timeoutDuration) {
          log.info("Peer %s::%s timed out\n", g.name, p.name);
          if (p.active) {
            g.needsUpdate = true;
          }
          i = g.peers.erase(i);
        } else {
          ++i;
        }
      }
      auto mintime = std::chrono::seconds(2);
      if (g.needsUpdate && (now - g.lastUpdate >= mintime)) {
        log.info("Initiating update of group %s\n", g.name);
        ++g.updateCount;
        g.lastUpdate = now;
        g.needsUpdate = false;
        uint32_t syncId = nextSyncId++;
        if (syncId == 0) {
          syncId = nextSyncId++;
        }
        g.syncId = syncId;
        for ([[maybe_unused]] auto& [pname, p] : g.peers) {
          p.syncFuture = call<std::pair<uint32_t, int32_t>>(pname, "GroupService::sync", g.name, syncId);
        }
        syncSet.insert(&g);
      }
    }
  }
};

struct Broker {

  std::shared_ptr<rpc::Rpc> rpc;
  BrokerService* brokerService = nullptr;

  Broker(std::shared_ptr<rpc::Rpc> rpc) : rpc(std::move(rpc)) {
    brokerService = this->rpc->getService<BrokerService>("BrokerService");
  }

  Broker() : rpc(std::make_shared<rpc::Rpc>()) {
    rpc->setName("broker");
    brokerService = rpc->getService<BrokerService>("BrokerService");
  }

  void setName(std::string name) {
    rpc->setName(name);
  }

  void listen(std::string address) {
    rpc->listen(address);
  }

  void update() {
    brokerService->update();
  }
};

} // namespace moolib
