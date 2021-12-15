# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

import pytest

import moolib


class MyValue:
    __slots__ = ["value"]

    def __init__(self, value):
        self.value = value


class MyClass:
    __slots__ = ["foo", "bar"]

    def __init__(self):
        self.foo = 42
        self.bar = {"key": MyValue(random.getrandbits(18))}


ADDRESS = "127.0.0.1:4422"


class TestMoolibPickle:
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

    def test_pickle_hello(self, host, client):
        def hello(message):
            print("Got hello: ", message.foo, message.bar["key"].value)
            return (
                "this is a response to message '" + str(message.foo) + "'",
                message.bar["key"].value,
            )

        host.define("hello", hello)

        for _ in range(10):
            input = MyClass()

            print("sending ", input.foo, input.bar["key"].value)

            msg, value = client.sync("host", "hello", input)
            print("sync: ", msg, value)
            assert value == input.bar["key"].value
