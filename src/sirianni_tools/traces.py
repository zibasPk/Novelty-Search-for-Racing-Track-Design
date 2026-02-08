"""Extract telemetry traces from simulation logs."""

import csv
import os
import logging
from utils import dynamicsLogColumns


__author__ = "Milo Brontesi"

def get_trace(folder, track_name, traced_column_idx, consider_laps = [2]):
    """
    Extracts telemetry_data and the position along the track from the dynamics log file.
    
    Args:
        folder: Path to the directory containing logs.
        track_name: Name of the track (prefix of the log file).
        traced_column_idx: Index of the column to extract from the dynamics log.
        consider_laps: List of lap numbers to consider for the trace.
    Returns:
        List[Tuple[float, float]]: A list of (telemetry_data, distance) tuples.
    """
    
    cols = dynamicsLogColumns
    
    trace = []
    
    try:
        # Find dynamics files in the folder (matching format: trackname_*_dynamics.csv)
        dyn_files = sorted([f for f in os.listdir(folder) 
                          if f.startswith(track_name) and f.endswith('_dynamics.csv')])
        
        if not dyn_files:
            return []

        # Use the last file found (consistent with blocks.py behavior)
        filepath = os.path.join(folder, dyn_files[-1])
        
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # Ensure row has enough columns
                if len(row) > 11:
                    try:
                        telemetry_data = float(row[traced_column_idx])
                        dist = float(row[cols.lap_distance.value])
                        lap = int(row[cols.lap.value])
                        if lap not in consider_laps:
                            continue
                        trace.append((telemetry_data, dist))
                    except ValueError:
                        continue
                        
    except Exception as e:
        logging.warning(f"Error extracting telemetry/distance trace: {e}")

    return trace





