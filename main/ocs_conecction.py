"""Small example OSC client

This program sends 10 random values between 0.0 and 1.0 to the /filter address,
waiting for 1 seconds between each value.
"""
import argparse
import random
import time

from pythonosc import udp_client

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ip", default="192.168.1.33",
      help="The ip of the OSC server")
  parser.add_argument("--port", type=int, default=7001,
      help="The port the OSC server is listening on")
  args = parser.parse_args()

  client = udp_client.SimpleUDPClient(args.ip, args.port)
  #client.send_message("/interaction", ['2023-07-17', '10:11:25', 2, 1, 0, 'sad','woman',31])
  client.send_message("/Metrics/DeepFace", ['2023-07-17', '10:11:25', 2, 1, 0, 'sad','woman',31])
  print("send Click")
  time.sleep(1)
