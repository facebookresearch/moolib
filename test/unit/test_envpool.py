# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import pytest

import torch
import moolib


class Env:
    def __init__(self):
        self.n = None

    def step(self, action):
        if action == 0:
            pass
        elif action == 1:
            self.n *= 2
        elif action == 2:
            self.n /= 2
        else:
            raise RuntimeError("bad action")
        obs = {"n": self.n}
        done = self.n.sum() < 1
        reward = abs(self.n.sum() - 4)
        return obs, reward, done

    def reset(self):
        self.n = torch.ones(4, 4)
        self.n[0][0] = 4.0
        self.n[3][1] = 0.5
        self.n[1][2] = 0.25
        return {"n": self.n}


class TestMoolibpEnvPool:
    def test_envpool(self):
        bs = 32
        envs = moolib.EnvPool(Env, batch_size=bs, num_batches=2, num_processes=4)

        with pytest.raises(RuntimeError):
            envs.step(0, torch.zeros(bs))
        with pytest.raises(RuntimeError):
            envs.step(0, torch.zeros(bs + 1).long())

        z = torch.zeros(bs).long()

        initial = torch.ones(4, 4)
        initial[0][0] = 4.0
        initial[3][1] = 0.5
        initial[1][2] = 0.25
        initial = initial.expand(bs, 4, 4)

        obs = envs.step(batch_index=0, action=z).result()
        assert obs["n"].equal(initial)
        fut0 = envs.step(batch_index=0, action=z + 1)
        fut1 = envs.step(batch_index=1, action=z)
        obs = fut0.result()
        assert obs["n"].equal(initial * 2)
        obs = fut1.result()
        assert obs["n"].equal(initial)
        obs = envs.step(batch_index=0, action=z + 2).result()
        assert obs["n"].equal(initial)

        states = [initial.clone(), initial.clone()]

        for _ in range(100):
            index = random.randint(0, 1)
            action = torch.randint(0, 3, [bs])
            s = states[index]
            obs = envs.step(index, action).result()
            reward = obs["reward"]
            done = obs["done"]
            for i in range(bs):
                if action[i] == 1:
                    s[i] *= 2
                elif action[i] == 2:
                    s[i] /= 2
                r = abs(s[i].sum() - 4)
                d = s[i].sum() < 1
                if d:
                    s[i] = initial[i]
                assert d == done[i]
                assert s[i].equal(obs["n"][i])
                assert r == reward[i]
