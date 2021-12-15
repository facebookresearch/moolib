# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import warnings

import torch
import pytest

import moolib

ADDRESS = "127.0.0.1:4422"


class TestMoolibSpeeds:
    @pytest.fixture
    def host(self):
        host = moolib.Rpc()
        host.set_name("host")
        host.listen(ADDRESS)
        yield host

    @pytest.fixture
    def client(self, host):
        client = moolib.Rpc()
        client.set_name("client")
        client.connect(ADDRESS)
        yield client

    def test_sync(self, host, client):
        weights = torch.randn(4096, 4096)

        def linear(inputs):
            return (weights * inputs).sum(-1)

        host.define("linear", linear)

        client.set_timeout(60)

        inputs = torch.randn(16, 4096)

        client.sync("host", "linear", inputs.unsqueeze(1))

    def test_sync_noop_speed(self, host, client):
        def noop():
            pass

        host.define("noop", noop)

        client.set_timeout(60)

        iterations = 128

        start = time.time()
        for _ in range(iterations):
            client.sync("host", "noop")
        t = time.time() - start

        print(
            "%d iterations took %f sec (%.1f per sec)" % (iterations, t, iterations / t)
        )

        if iterations / t < 1000:
            warnings.warn(f"Very slow iteration speed: {iterations / t}")

    def test_async_noop_speed(self, host, client):
        def noop():
            pass

        host.define("noop", noop)

        futures = []
        start = time.time()
        for _ in range(2000):
            futures.append(client.async_("host", "noop"))
        for i in futures:
            i.result()
        t = time.time() - start
        print("noop x%d time %g (%g/s)" % (len(futures), t, len(futures) / t))

        if len(futures) / t < 500:
            warnings.warn(f"Very slow iteration speed: {len(futures) / t}")

    def test_async_vs_local(self, host, client):
        weights = torch.randn(4096, 4096)

        def linear(inputs):
            return (weights * inputs).sum(-1)

        host.define("linear", linear)

        inputs = torch.randn(16, 4096)
        for _ in range(2):
            start = time.time()
            local_result = sum(
                linear(inputs[i]).sum().item() for i in range(inputs.size(0))
            )
            print("base time ", time.time() - start)

        for _ in range(4):
            futures = []
            start = time.time()
            for i in range(inputs.size(0)):
                futures.append(client.async_("host", "linear", inputs[i]))
            result = sum(i.result().sum().item() for i in futures)
            print("async time ", time.time() - start)
            assert abs(result - local_result) < 0.1

        host.debug_info()
        client.debug_info()
