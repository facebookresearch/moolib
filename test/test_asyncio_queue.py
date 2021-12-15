# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import moolib
import time
import torch

import asyncio


async def process(queue, callback):
    try:
        while True:
            return_callback, args, kwargs = await queue
            if args and kwargs:
                retval = callback(*args, **kwargs)
            elif args:
                retval = callback(*args)
            elif kwargs:
                retval = callback(**kwargs)
            else:
                retval = callback()
            return_callback(retval)
    except asyncio.CancelledError:
        print("process cancelled")
        pass
    except Exception as e:
        print(e)
        raise


async def main():

    loop = asyncio.get_running_loop()

    localAddr = "127.0.0.1:4411"
    # localAddr = "shm://testtest"

    client = moolib.Rpc()
    host = moolib.Rpc()

    client.set_name("client")
    client.set_timeout(1)

    def client_hello(*args):
        print("client got hello:", (*args,))

    client.define("client hello", client_hello)

    def hello(message):
        print("Got hello: ", message)
        return "this is a response to message '" + message + "'"

    def hello_deferred(callback, message):
        print("Got hello (deferred): ", message)
        callback("this is the deferred response to " + message)

    host.define_deferred("hello deferred", hello_deferred)

    def wrap_define(self, name, func):
        loop.create_task(process(self.define_queue(name), func))

    wrap_define(host, "hello", hello)

    host.set_name("host")
    host.listen(localAddr)

    client.connect(localAddr)

    print(client.sync("host", "hello deferred", "sync test"))
    print(client.sync("host", "hello deferred", message="named argument"))

    foo = client.async_("host", "hello", "this is a message from client")
    response = await foo
    print("Got response: ", response)

    try:
        # sync will block and time out since we can't process the queue
        print("sync: ", client.sync("host", "hello", "sync test"))
        raise AssertionError()
    except RuntimeError as e:
        print(e)

    print("async: ", await client.async_("host", "hello", "async test"))

    weights = torch.randn(4096, 4096)

    def linear(input):
        return (weights * input).sum(-1)

    def noop():
        pass

    wrap_define(host, "linear", linear)
    wrap_define(host, "noop", noop)

    client.set_timeout(60)

    input = torch.randn(16, 4096)

    await client.async_("host", "linear", input.unsqueeze(1))

    for _ in range(128):
        await client.async_("host", "noop")

    for _ in range(4):
        futures = []
        start = time.time()
        for _ in range(10000):
            futures.append(client.async_("host", "noop"))
        for i in futures:
            await i
        t = time.time() - start
        print("noop x%d time %g (%g/s)" % (len(futures), t, len(futures) / t))

    for _ in range(2):
        start = time.time()
        local_result = sum(linear(input[i]).sum().item() for i in range(input.size(0)))
        print("base time ", time.time() - start)

    for _ in range(4):
        futures = []
        start = time.time()
        for i in range(input.size(0)):
            futures.append(client.async_("host", "linear", input[i]))
        result = 0
        for i in futures:
            result += (await i).sum().item()
        # why does this not work ?
        # result = sum((await i).sum().item() for i in futures)
        print("async time ", time.time() - start)
        assert abs(result - local_result) < 0.1

    host.debug_info()
    client.debug_info()


try:
    asyncio.run(main())
except:
    import traceback

    traceback.print_exc()
