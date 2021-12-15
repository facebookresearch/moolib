# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import moolib
import time
import torch
import random

moolib.set_log_level("debug")

current_reduce_size = 0
current_reduce_done = 0
current_reduce_sum = 0


class Client:
    def __init__(self, broker_addr, index):
        self.rpc = moolib.Rpc()
        self.rpc.set_name("client %d" % index)
        self.rpc.set_timeout(20)
        self.rpc.connect(broker_addr)
        self.group = None
        self.index = index
        self.reduce = None
        self.sum = None
        self.reduce_counter = 0

    def update(self):

        if self.group is None:
            self.group = moolib.Group(self.rpc, "test group")
            self.group.set_timeout(1)
            self.group.set_sort_order(self.index)
        elif random.random() < 0.5 or True:
            updated = self.group.update()

            if updated:
                print(
                    "group '%s', sync id %#x, members %s"
                    % (self.group.name(), self.group.sync_id(), self.group.members())
                )

        if self.group.active():
            if self.reduce is None:
                self.start_reduce()
            else:
                result = None

                global current_reduce_size
                global current_reduce_done
                global current_reduce_sum

                try:
                    if self.reduce.done():
                        # although the result is available in self.tensor,
                        # calling .result() here is necessary to raise any errors
                        result = self.reduce.result()
                except RuntimeError as e:
                    current_reduce_size = 0
                    current_reduce_done = 0
                    print(e)
                    self.reduce = None

                if result is not None:
                    self.sum = result.sum().item()
                    print("reduced to sum ", self.sum)
                    self.reduce = None
                    if current_reduce_size == current_reduce_done:
                        current_reduce_done = 1
                        current_reduce_size = len(self.group.members())
                        current_reduce_sum = self.sum
                        print("New reduction of size %d" % current_reduce_size)
                    else:
                        current_reduce_done += 1
                        if abs(self.sum - current_reduce_sum) > 0.01:
                            raise RuntimeError(
                                "Reduce sum mismatch, got %g, expected %g"
                                % (self.sum, current_reduce_sum)
                            )
                        print(
                            "Reduce %d/%d done"
                            % (current_reduce_done, current_reduce_size)
                        )

    def start_reduce(self):
        self.tensor = torch.randn(64, 64)
        print("input sum ", self.tensor.sum().item())
        self.reduce = self.group.all_reduce("test reduce", self.tensor)


def main():

    localAddr = "127.0.0.1:4411"

    broker_rpc = moolib.Rpc()
    broker_rpc.set_name("broker")
    broker = moolib.Broker(broker_rpc)
    broker_rpc.listen(localAddr)

    clients = []
    for i in range(4):
        clients.append(Client(localAddr, i))

    for _ in range(30):

        for client in clients:
            client.update()

        broker.update()

        time.sleep(0.1)

    # assert clients[0].group.members() == ["client 0", "client 1", "client 2", "client 3"]
    # del clients[1]

    for _ in range(300):

        for client in clients:
            client.update()

        broker.update()

        time.sleep(0.1)

    print("All done")


main()
