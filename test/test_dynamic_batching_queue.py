# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import time
import traceback

import torch

import moolib


async def process(que, callback):
    try:
        while True:
            ret_cb, args, kwargs = await que
            if args and kwargs:
                ret = callback(*args, **kwargs)
            elif args:
                ret = callback(*args)
            elif kwargs:
                ret = callback(**kwargs)
            else:
                ret = callback()
            ret_cb(ret)
    except asyncio.CancelledError:
        print("[Server] process cancelled")
        pass
    except Exception as e:
        print(e)
        raise


async def main():
    addr = "127.0.0.1:4411"
    timeout = 60

    num_tests = 200
    num_benchmarks = 10000
    dim = 128
    linear = torch.nn.Linear(dim, dim)

    loop = asyncio.get_running_loop()

    server = moolib.Rpc()
    server.set_name("server")
    server.set_timeout(timeout)

    verbose = True

    def run_linear(x, info):
        if verbose:
            print(f"[Linear] batch_size = {x.size()[0]}")

        with torch.no_grad():
            return linear(x), info

    loop.create_task(process(server.define_queue("linear"), run_linear))
    loop.create_task(
        process(
            server.define_queue("batch_linear", batch_size=100, dynamic_batching=True),
            run_linear,
        )
    )
    server.listen(addr)

    client = moolib.Rpc()
    client.set_name("client")
    client.set_timeout(timeout)
    client.connect(addr)

    x_list = [torch.randn(dim) for _ in range(num_tests)]
    y_list = [linear(x) for x in x_list]

    futs = []
    for i, x in enumerate(x_list):
        futs.append(
            client.async_("server", "batch_linear", x, info=dict(index=[i, i + 1]))
        )
    for i, fut in enumerate(futs):
        y, info = await fut
        assert torch.allclose(y, y_list[i], rtol=1e-5, atol=1e-6)
        assert len(info) == 1
        assert info["index"] == [i, i + 1]

    verbose = False
    futs1 = []
    futs2 = []

    t0 = time.time()
    for _ in range(num_benchmarks):
        futs1.append(client.async_("server", "linear", x_list[0], 0))
    for fut in futs1:
        await fut
    t1 = time.time()
    for _ in range(num_benchmarks):
        futs2.append(client.async_("server", "batch_linear", x_list[0], 0))
    for fut in futs2:
        await fut
    t2 = time.time()

    print(f"[Benchmark] without batching time: {t1 - t0} seconds")
    print(f"[Benchmark] dynamic batching time: {t2 - t1} seconds")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except:
        traceback.print_exc()
