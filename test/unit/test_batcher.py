# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

import torch
import moolib


class TestMoolibBatcher:
    def test_batcher(self):
        for _ in range(256):
            size = random.randint(1, 20)
            dim = random.randint(0, 2)
            dims = random.randint(dim + 1, dim + 2)

            n = random.randint(20, 100)
            shape = [random.randint(1, 4) for _ in range(dims)]

            batcher = moolib.Batcher(size=size, dim=dim)

            inputs = []
            for _ in range(n):
                input = torch.randn(shape)
                inputs.append(input.clone())
                batcher.stack(input)
                assert input.equal(inputs[-1])
                if not batcher.empty():
                    batched = batcher.get()
                    stacked = torch.stack(inputs, dim=dim)
                    assert batched.equal(stacked)
                    inputs = []

            batcher = moolib.Batcher(size=size, dim=dim)

            inputs = []
            for _ in range(n):
                input = torch.randn(shape)
                inputs.append(input.clone())
                batcher.cat(input)
                assert input.equal(inputs[-1])
                if not batcher.empty():
                    batched = batcher.get()
                    catted = torch.cat(inputs, dim=dim)
                    overflow = catted.narrow(dim, size, catted.size(dim) - size)
                    catted = catted.narrow(dim, 0, size)
                    assert batched.equal(catted)
                    inputs = []
                    if overflow.size(dim) > 0:
                        inputs.append(overflow)
