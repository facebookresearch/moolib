# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import moolib
import time
import torch


def main():

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

    host.define("hello", hello)

    host.set_name("host")
    host.listen(localAddr)

    client.connect(localAddr)

    future = client.async_("host", "hello", "this is a message from client")

    response = future.result()

    print("Got response: ", response)

    print("sync: ", client.sync("host", "hello", "sync test"))

    if True:

        def helloCallback(response, error):
            print("Callback response: ", response)
            print("Callback error: ", error)
            assert error is None

        client.async_callback(
            "host", "hello", helloCallback, "this is a message through async_callback"
        )

        try:
            future = client.async_("nowhere", "hello", "this is a message to nowhere")
            response = future.result()
            print("Response from nowhere: ", response)
            raise AssertionError()
        except Exception as e:
            print(e)

        try:
            client.sync("host", "non-existant function")
            raise AssertionError()
        except Exception as e:
            print(e)

        del host

        try:
            client.sync("host", "hello", "is host dead?")
            raise AssertionError()
        except Exception as e:
            print(e)

        host = moolib.Rpc()
        host.set_name("host")
        host.listen(localAddr)
        host.set_timeout(30)

        # host does not connect to client, so this would normally not work
        # but since client lost connection to host, it should reconnect automatically
        print(host.sync("client", "client hello", "I", "am", "back!"))

    weights = torch.randn(4096, 4096)

    def linear(input):
        return (weights * input).sum(-1)

    def noop():
        pass

    host.define("linear", linear)
    host.define("noop", noop)

    client.set_timeout(60)

    input = torch.randn(16, 4096)

    client.sync("host", "linear", input.unsqueeze(1))

    for _ in range(128):
        client.sync("host", "noop")

    for _ in range(4):
        futures = []
        start = time.time()
        for _ in range(10000):
            futures.append(client.async_("host", "noop"))
        for i in futures:
            i.result()
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
        result = sum(i.result().sum().item() for i in futures)
        print("async time ", time.time() - start)
        assert abs(result - local_result) < 0.1

    host.debug_info()
    client.debug_info()


main()
