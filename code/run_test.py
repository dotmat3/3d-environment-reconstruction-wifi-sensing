import threading
import time
import os
import multiprocessing
import signal
import subprocess
import io
import csv

import argparse

parser = argparse.ArgumentParser(description="Script to run ESP experiments")
parser.add_argument("-ne", "--num-exp", type=int, help="Number of experiments to perform", default=1)
parser.add_argument("-w", "--wait", type=int, help="Number of seconds to wait before starting the experiment", default=0)
parser.add_argument("-p", "--packets", type=int, help="Number of packets to collect during the experiment", required=True)
parser.add_argument("output", type=str, help="CSV output file")

args = parser.parse_args()

seconds_before_starting = args.wait
num_packets = args.packets
filename = args.output
num_exp = args.num_exp

for idx_exp in range(num_exp):
    print(f"Starting experiment #{idx_exp}")

    if seconds_before_starting != 0:
        print("Waiting", seconds_before_starting, "seconds before starting...")
        time.sleep(seconds_before_starting)

    print("Starting")

    process = subprocess.Popen(["idf.py", "-p COM3", "monitor"], shell=True, text=True, stdout=subprocess.PIPE)
    print("Spawned")
    start = time.time()
    count = 0

    f = open(filename.replace("#", idx_exp), "w")

    print(f"{count}/{num_packets} packets", end="\r")

    while True:
        line = process.stdout.readline()
        if not line:
            break
        
        if line.startswith("CSI_DATA"):
            count += 1

            f.write(line.strip() + "\n")
            if count % 10 == 0:
                print(f"{count}/{num_packets} packets", end="\r")

        if count == num_packets:
                break

    print(f"{count}/{num_packets} packets")

    p = subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=process.pid), stdout=subprocess.PIPE)
    p.communicate()
    print("Terminated")

    f.close()
    print("Result saved in", filename)
