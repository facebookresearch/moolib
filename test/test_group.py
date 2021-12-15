# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import moolib
import time

moolib.set_log_level("info")


class Client:
    def __init__(self, broker_addr, index):
        self.rpc = moolib.Rpc()
        self.name = "client %d" % index
        self.rpc.set_name(self.name)
        self.rpc.set_timeout(2)
        self.rpc.define("hello", self.hello)
        self.rpc.connect(broker_addr)
        self.group = None
        self.index = index

    def hello(self, str):
        print("%s received hello: %s" % (self.name, str))

    def update(self):

        if self.group is None:
            self.group = moolib.Group(self.rpc, "test group")
            self.group.set_timeout(1)
            self.group.set_sort_order(-self.index)
        else:
            updated = self.group.update()

            if updated:
                print(
                    "group '%s', sync id %#x, members %s"
                    % (self.group.name(), self.group.sync_id(), self.group.members())
                )
                for n in self.group.members():
                    if n != self.name:
                        self.rpc.async_(n, "hello", "hello from " + self.name)


def main():

    localAddr = "127.0.0.1:4411"
    # localAddr = "shm://testtest"

    broker_rpc = moolib.Rpc()
    broker_rpc.set_name("broker")
    broker = moolib.Broker(broker_rpc)
    broker_rpc.listen(localAddr)

    clients = []
    for i in range(4):
        clients.append(Client(localAddr, i))

    for _ in range(10):

        for client in clients:
            client.update()

        broker.update()

        time.sleep(0.1)

    assert clients[0].group.members() == [
        "client 3",
        "client 2",
        "client 1",
        "client 0",
    ]

    del clients[2]

    for _ in range(20):

        for client in clients:
            client.update()

        broker.update()

        time.sleep(0.1)

    assert clients[0].group.members() == ["client 3", "client 1", "client 0"]


main()
