# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
from unittest import mock

from examples import a2c


class TestA2CExample:
    def test_single_node_training(self, num_steps=40000):
        items = collections.deque(maxlen=20)

        def log_func(**kwargs):
            items.append(kwargs)

        with mock.patch.object(a2c, "log_to_file", log_func):
            a2c.train(num_steps)

        low_return_items = []

        for index, item in enumerate(items):
            max_step_offset = (
                2 * (len(items) - index) * a2c.BATCH_SIZE * a2c.ROLLOUT_LENGTH
            )
            assert item["step"] > num_steps - max_step_offset
            if item["mean_episode_return"] < 100:
                low_return_items.append((index, item))
            assert -1 < item["entropy_loss"] < 0

        assert len(low_return_items) / len(items) < 0.5
