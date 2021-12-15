# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import moolib
import time
import torch

import asyncio

moolib.set_log_level("verbose")

localAddr = "127.0.0.1:4411"
# localAddr = "shm://testest"

loop = asyncio.get_event_loop()

terminate = False


class Group:
    def __init__(self, rpc, group_name):
        self.group = moolib.Group(rpc, group_name)
        self.group.set_timeout(2)
        self.rpc = rpc
        self.my_name = rpc.get_name()
        self.group_name = group_name
        self.members = []

    def update(self):
        if self.group.update():
            self.members = self.group.members()

    def active(self):
        return self.group.active()

    async def wait_for_active(self):
        while not self.active():
            await asyncio.sleep(0.25)

    async def all_reduce(self, name, tensor):
        return await self.group.all_reduce(name, tensor)


async def keepalive(obj):
    try:
        while True:
            obj.update()
            await asyncio.sleep(0.25)
    except asyncio.CancelledError:
        print("keepalive cancelled")
        pass


inputsum = {}
reducesum = None


async def client(index, n_clients):
    rpc = moolib.Rpc()
    rpc.set_name("client %d" % index)
    rpc.set_timeout(2)
    rpc.connect(localAddr)

    group = Group(rpc, "test group")

    group_keepalive = loop.create_task(keepalive(group))

    my_name = rpc.get_name()

    global inputsum
    global reducesum

    n = 0

    while not terminate:
        try:
            await group.wait_for_active()

            tensor = torch.randn(64, 64)
            print("input sum ", tensor.sum().item())
            inputsum[index] = tensor.sum().item()
            try:
                start = time.time()
                localsum = await group.all_reduce("test reduce", tensor)
                print("allreduce took %g" % (time.time() - start))
            except RuntimeError as e:
                print(e)
                continue

            mysum = localsum.sum().item()
            print("reduce %d done -> sum %g" % (index, mysum))

            if len(inputsum) == n_clients:
                actualsum = sum(v for k, v in inputsum.items())
                for k, v in inputsum.items():
                    print(k, v)
                inputsum = {}

                if abs(mysum - actualsum) > 1e-2:
                    raise RuntimeError(
                        "sum mismatch: my sum is %g, real sum is %g"
                        % (mysum, actualsum)
                    )
                reducesum = actualsum
            elif abs(mysum - reducesum) > 1e-2:
                raise RuntimeError(
                    "sum mismatch: my sum is %g, should be %g" % (mysum, actualsum)
                )

            n += 1

            # await asyncio.sleep(2)

        except asyncio.CancelledError:
            print("client cancelled!")
            break

    del group_keepalive
    print(my_name, "normal exit :)")


async def broker():
    broker_rpc = moolib.Rpc()
    broker_rpc.set_name("broker")
    broker = moolib.Broker(broker_rpc)
    broker_rpc.listen(localAddr)
    t = time.time()
    while not terminate:
        now = time.time()
        print("Time since last broker update: %g" % (now - t))
        t = now
        broker.update()
        await asyncio.sleep(0.25)
    print("broker done!")


async def wait(tasks, timeout):
    done, pending = await asyncio.wait(
        tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
    )
    for i in done:
        i.result()
    return pending


async def main():
    global terminate

    broker_task = loop.create_task(broker())
    await wait([broker_task], 0.25)

    # await wait([broker_task], 120)

    n_clients = 11

    clients = []
    for i in range(n_clients):
        clients.append(loop.create_task(client(i, n_clients)))

    await wait([broker_task, *clients], 45)
    terminate = True
    for i in clients:
        i.cancel()
    await asyncio.gather(broker_task, *clients)

    print("All done")


try:
    loop.run_until_complete(main())
except:
    import traceback

    traceback.print_exc()
