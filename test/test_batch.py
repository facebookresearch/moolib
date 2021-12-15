# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import moolib
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

    host = moolib.Rpc()

    bs = 8

    def hello(message, tensor):
        print("Got hello: ", message, tensor.shape)
        for i in range(tensor.size(0)):
            print("tensor[%d].sum() is %g" % (i, tensor[i].sum().item()))
        r = tensor.flatten(1).sum(1)
        print("returning ", r.shape)
        return "this is a response to message '" + message + "'", r

    host.define("hello", hello, batch_size=bs)

    def hello_deferred(callback, message, tensor):
        callback(hello(message, tensor))

    host.define_deferred("hello deferred", hello_deferred, batch_size=bs)

    def wrap_define(self, name, func, batch_size=None):
        return loop.create_task(
            process(self.define_queue(name, batch_size=batch_size), func)
        )

    wrap_define(host, "hello queue", hello, batch_size=bs)

    host.set_name("host")
    host.listen(localAddr)

    clients = []
    for _ in range(40):
        client = moolib.Rpc()
        client.set_timeout(10)
        client.connect(localAddr)
        clients.append(client)

    for i in range(21):
        futures = []
        fn = ["hello", "hello deferred", "hello queue"][i % 3]
        for c in clients:
            t = torch.randn(2, 3)
            print("calling with tensor sum %g" % t.sum().item())
            futures.append(c.async_("host", fn, "wee " + fn, t))

        for f in futures:
            try:
                print(await f)
            except Exception as e:
                print(e)
                raise

    # host.debug_info()


try:
    asyncio.run(main())
except:
    import traceback

    traceback.print_exc()
