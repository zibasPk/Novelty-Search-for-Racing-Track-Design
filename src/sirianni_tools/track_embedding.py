

import csv
import os

import utils

DEFAULT_LAP = 2


def generate(trackName):
    """Generate track embedding from dynamics logs."""
    log_files = sorted([f for f in os.listdir(utils.torcsLogPath)
                        if f.startswith(trackName) and f.endswith('_dynamics.csv')])

    segments = []

    segCounter = 0

    with open(os.path.join(utils.torcsLogPath, log_files[-1]), 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            if len(row) != 26:
                continue
            lap = int(row[12])

            if lap != DEFAULT_LAP:
                continue
            idx = segCounter
            speed = float(row[5])
            steer = float(row[22])
            accel = float(row[23])
            brake = float(row[24])
            gear = int(row[25])
            angle_to_track = float(row[9])

            segments.append([idx, speed, steer, accel, brake, gear, angle_to_track])
            segCounter += 1

    return segments
