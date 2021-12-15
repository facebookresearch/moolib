# Copyright (c) Facebook, Inc. and its affiliates.
import moolib
import argparse
import time

# TODO: Can moolib choose a port for us?
DEFAULT_PORT = 4431

parser = argparse.ArgumentParser(description="A script to run a moolib broker")

parser.add_argument(
    "address",
    nargs="?",
    default="0.0.0.0:%i" % DEFAULT_PORT,
    type=str,
    metavar="addr:port",
    help="Broker server address to listen on.",
)


def main():
    FLAGS = parser.parse_args()

    broker_rpc = moolib.Rpc()
    broker_rpc.set_name("broker")
    broker = moolib.Broker(broker_rpc)
    broker_rpc.listen(FLAGS.address)

    print("Broker listening at %s" % FLAGS.address)

    try:
        while True:
            broker.update()
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
