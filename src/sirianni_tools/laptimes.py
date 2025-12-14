
"""Analyze lap times from simulation logs."""

import os
import logging
from collections import defaultdict

__author__ = "Jacopo Sirianni"
__copyright__ = "Copyright 2015-2016, Jacopo Sirianni"
__license__ = "GPL"
__email__ = "jacopo.sirianni@mail.polimi.it"

def compute_lap_times(folder, log_list, track_length):
    """
    Computes the average lap time for each lap number across all drivers.
    
    Args:
        folder: Path to the directory containing logs.
        log_list: List of log filenames.
        track_length: Float representing the total length of the track.
        
    Returns:
        Dict[str, float]: Mapping from lap number (as string) to average time in seconds.
                          Example: {'1': 85.4, '2': 82.1, '3': 81.5}
    """
    # Dictionary mapping lap_number (int) -> list of times (float)
    lap_durations = defaultdict(list)
    
    # Minimum reasonable lap time to filter glitches (e.g., immediate wrap-around)
    MIN_LAP_TIME = 10.0
    
    for log in log_list:
        filepath = os.path.join(folder, log)
        try:
            # State per driver: { 'last_dist': float, 'lap_start_time': float, 'current_lap': int }
            driver_data = {}
            
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    
                    # Ensure we have enough columns
                    # Index 0: Timestamp, Index 1: Driver, Index 11: Distance from start
                    if len(parts) < 12: 
                        continue
                    
                    try:
                        timestamp = float(parts[0])
                        driver = parts[1]
                        dist = float(parts[11])
                    except ValueError:
                        continue
                        
                    # Skip pre-race countdown times
                    if timestamp < 0: 
                        continue
                    
                    if driver not in driver_data:
                        # Initialize driver state
                        driver_data[driver] = {
                            'last_dist': dist,
                            'lap_start_time': timestamp,
                            'current_lap': 1
                        }
                    
                    curr = driver_data[driver]
                    
                    # Detect Lap Finish: Distance wraps around from near TrackLength to near 0
                    # Using 80% and 20% thresholds to identify the wrap-around point robustly
                    if curr['last_dist'] > (track_length * 0.8) and dist < (track_length * 0.2):
                        lap_time = timestamp - curr['lap_start_time']
                        
                        if lap_time > MIN_LAP_TIME:
                            lap_durations[curr['current_lap']].append(lap_time)
                            
                        # Reset start time and increment lap counter
                        curr['lap_start_time'] = timestamp
                        curr['current_lap'] += 1
                        
                    curr['last_dist'] = dist
            
        except Exception as e:
            print(f"Warning: Error computing laptimes for {log}: {e}")
            
    # Compute averages and return as a dictionary
    avg_times = {}
    for lap_num in sorted(lap_durations.keys()):
        times = lap_durations[lap_num]
        if times:
            # Use string keys for better JSON compatibility
            avg_times[str(lap_num)] = sum(times) / len(times)
            
    return avg_times