import moolib
import argparse
import time

parser = argparse.ArgumentParser(description="hello world")

parser.add_argument(
    "address", default="", type=str, help="Broker server address to listen on"
)

flags = parser.parse_args()

broker_rpc = moolib.Rpc()
broker_rpc.set_name("broker")
broker = moolib.Broker(broker_rpc)
broker_rpc.listen(flags.address)

while True:
    broker.update()
    time.sleep(0.25)
