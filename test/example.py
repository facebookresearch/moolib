# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import moolib


def printFunction(str):
    print(str)
    return 42


host = moolib.Rpc()
host.set_name("host")
host.define("print", printFunction)
host.listen("127.0.0.1:1234")

client = moolib.Rpc()
client.connect("127.0.0.1:1234")

future = client.async_("host", "print", "hello world")
print(future.get())


client.define("sum", sum)
print(host.sync(client.get_name(), "sum", [1, 2, 3, 4], 10))
