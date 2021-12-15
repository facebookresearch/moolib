# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import moolib.broker


class TestBrokerScript:
    def test_has_main(self):
        assert hasattr(moolib.broker, "main")

    @mock.patch("sys.argv", ["broker"])
    def test_run(self):
        with mock.patch("moolib.Broker") as MockBroker:
            instance = MockBroker.return_value
            instance.update.side_effect = [
                None,
                None,
                None,
                KeyboardInterrupt("Enough"),
            ]

            moolib.broker.main()

            MockBroker.assert_called_once()
            assert instance.update.call_count == 4
