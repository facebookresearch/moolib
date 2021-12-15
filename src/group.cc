/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "group.h"

namespace moolib {

struct GroupImpl {

  void connect(std::string brokerAddress, std::string groupName);
  void ping();
  bool connected();
  std::vector<std::string> members();
};

Group::Group() {
  impl = std::make_unique<GroupImpl>();
}

Group::~Group() {}

void Group::connect(std::string brokerAddress, std::string groupName) {
  impl->connect(brokerAddress, groupName);
}

void Group::ping() {
  impl->ping();
}

bool Group::connected() {
  return impl->connected();
}

std::vector<std::string> Group::members() {
  return impl->members();
}

} // namespace moolib
