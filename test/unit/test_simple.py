# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

import pytest

import moolib

ADDRESS = "127.0.0.1:4411"


class TestMoolib:
    def test_call_async_and_sync(self):
        client = moolib.Rpc()
        host = moolib.Rpc()

        client.set_name("client")
        client.set_timeout(1)

        num_calls = 0

        def hello(message):
            nonlocal num_calls
            num_calls += 1
            print("Got hello: ", message)
            return "this is a response to message '" + message + "'"

        host.define("hello", hello)

        host.set_name("host")
        host.listen(ADDRESS)

        client.connect(ADDRESS)

        message = "this is a message from client"
        future = client.async_("host", "hello", message)

        response = future.result()

        assert num_calls == 1
        assert response == hello(message)

        message2 = "sync test"
        assert client.sync("host", "hello", message2) == hello(message2)
        assert num_calls == 4

    def test_async_callback_and_unknown_peer(self):
        client = moolib.Rpc()
        host = moolib.Rpc()

        client.set_name("client")
        client.set_timeout(1)

        def hello(message):
            return "this is a response to message %s" % repr(message)

        host.define("hello", hello)
        host.set_name("host")
        host.listen(ADDRESS)
        client.connect(ADDRESS)

        num_calls = 0
        message = "this is a message through async_callback"

        def helloCallback(response, error):
            nonlocal num_calls
            num_calls += 1
            assert response == hello(message)
            assert error is None

        client.async_callback("host", "hello", helloCallback, message)

        future = client.async_("nowhere", "hello", "this is a message to nowhere")
        with pytest.raises(  # TODO: Should this be a RuntimeError?
            RuntimeError, match=re.escape("Call (nowhere::<unknown>) timed out")
        ):
            future.result()

        assert num_calls == 1

    def test_nonexistent_function_and_dead_host(self):
        client = moolib.Rpc()
        host = moolib.Rpc()

        client.set_name("client")
        client.set_timeout(1)

        called = False

        def client_hello(*args):
            nonlocal called
            called = True
            print("client got hello:", (*args,))

        client.define("client hello", client_hello)

        def hello(message):
            pass

        host.define("hello", hello)
        host.set_name("host")
        host.listen(ADDRESS)
        client.connect(ADDRESS)

        with pytest.raises(  # TODO: Should this be a RuntimeError?
            RuntimeError,
            match=re.escape(
                "RPC remote function host::'non-existent function' does not exist"
            ),
        ):
            client.sync("host", "non-existent function")

        del host

        with pytest.raises(
            # TODO: Why is this <unknown> and not "hello"?
            RuntimeError,
            match=re.escape("Call (host::<unknown>) timed out"),
        ):
            client.sync("host", "hello", "is host dead?")

        host = moolib.Rpc()
        host.set_name("host")
        host.listen(ADDRESS)
        host.set_timeout(30)

        # host does not connect to client, so this would normally not work
        # but since client lost connection to host, it should reconnect automatically
        # TODO: This call adds ~8 sec to the test.
        host.sync("client", "client hello", "I", "am", "back!")
        assert called
