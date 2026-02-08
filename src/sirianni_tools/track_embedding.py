

import csv
import os

from utils import torcsLogPath, dynamicsLogColumns


DEFAULT_LAP = 2

__author__ = "Milo Brontesi"

def generate(trackName):
    """Generate track embedding from dynamics logs."""
    cols = dynamicsLogColumns
    log_files = sorted([f for f in os.listdir(torcsLogPath)
                        if f.startswith(trackName) and f.endswith('_dynamics.csv')])

    segments = []

    segCounter = 0

    with open(os.path.join(torcsLogPath, log_files[-1]), 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            if len(row) != 26:
                continue
            lap = int(row[cols.lap.value])

            if lap != DEFAULT_LAP:
                continue
            idx = segCounter
            speed = float(row[cols.speed.value])
            steer = float(row[cols.steer.value])
            accel = float(row[cols.accel.value])
            brake = float(row[cols.brake.value])
            gear = int(row[cols.gear.value])
            right_border_dist = float(row[cols.right_border_dist.value])

            segments.append([idx, speed, steer, accel, brake, gear, right_border_dist])
            segCounter += 1

    return segments
