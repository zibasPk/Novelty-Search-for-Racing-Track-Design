#!/usr/bin/env python

"""Run track export or race simulations multiple times, then average analyses."""

import argparse
import os
import sys
import subprocess
import math
import json

import racegen
import utils

__author__ = "Jacopo Sirianni"
__copyright__ = "Copyright 2015-2016, Jacopo Sirianni"
__license__ = "GPL"
__email__ = "jacopo.sirianni@mail.polimi.it"


def run_track_export(folder_name):
    """Run a single lap with only trackexporter."""
    print("==> Running track export...")
    race_config = os.path.join(folder_name, "track_export.xml")
    
    # Generate config with only trackexporter
    racegen.generate_track_export_xml(race_config)
    
    cmd = f"{utils.torcsCommand} -r {os.path.join(os.getcwd(), race_config)}"
    subprocess.check_call(cmd, shell=True)
    os.remove(race_config)
   
def clear_logs():
    """Clear existing logs in the torcs log path."""
    log_path = utils.torcsLogPath
    if os.path.exists(log_path):
        for filename in os.listdir(log_path):
            file_path = os.path.join(log_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting log file {file_path}: {e}")   
  
def run_benchmark_sim(folder_name):
    """Run benchmark simulation with one bot for various measurement."""
    print("==> Running benchmark simulation...")  
    race_config = os.path.join(folder_name, "lap_num_sim.xml")
    
    # Generate config with only trackexporter
    racegen.generate_benchmark_xml(race_config)
    
    cmd = f"{utils.torcsCommand} -r {os.path.join(os.getcwd(), race_config)}"
    subprocess.check_call(cmd, shell=True)
    os.remove(race_config)
    
def run_lap_num_sim(folder_name, target_duration):
    """Run a single lap with only one car to decide number of laps."""
    print("==> Running lap number simulation...")
    race_config = os.path.join(folder_name, "lap_num_sim.xml")
    
    # Generate config with only trackexporter
    racegen.generate_benchmark_xml(race_config)
    
    cmd = f"{utils.torcsCommand} -r {os.path.join(os.getcwd(), race_config)}"
    subprocess.check_call(cmd, shell=True)
    os.remove(race_config)
    
    output = run_analysis(no_plots=True, json_output=True)
    # Parse output to find number of laps
    parsed_output = parse_analysis_json(output)
    lap_times = parsed_output.get("lap_times", 1)
    
    lap_time = None
    if isinstance(lap_times, dict) and '2' in lap_times:
        lap_time = lap_times['2'] 
    
    
    return math.ceil(target_duration / lap_time)

def run_embedding_simulation(folder_name):
    """Run two lap simulation (takes the second lap data) with one car for track embedding."""
    print("==> Running track embedding simulation...")
    race_config = os.path.join(folder_name, "embedding_sim.xml")
    
    # Generate config with only trackexporter
    racegen.generate_benchmark_xml(race_config)
    
    cmd = f"{utils.torcsCommand} -r {os.path.join(os.getcwd(), race_config)}"
    subprocess.check_call(cmd, shell=True)
    os.remove(race_config)
    
    output = run_analysis(no_plots=True, json_output=True)
    # Parse output to find number of laps
    parsed_output = parse_analysis_json(output)
    embedding = parsed_output.get("track_embedding", None)
    return embedding

def run_race_simulation(folder_name, num_laps, iteration=0, change_order=True):
    """Run main race simulation with all bots."""
    print(f"==> Running race simulation for {num_laps} laps...")
    race_config = os.path.join(folder_name, "race_sim.xml")
    
    # Generate config with all racing bots
    racegen.generate_race_xml(race_config, num_laps, iteration=iteration, change_order=change_order)
    
    cmd = f"{utils.torcsCommand} -r {os.path.join(os.getcwd(), race_config)}"
    subprocess.check_call(cmd, shell=True)
    print("Race simulation completed.")
    
def run_analysis(no_plots=True, json_output=True, embedding=False, trace=False):
    """
    Run analysis script on the logs, returning output as string.
    By default uses JSON output from analyze_simulations.py.
    """
    print("==> Analyzing logs...")
    cmd = [
        "python3",
        "/usr/local/lib/sirianni_tools/analyze_simulations.py",
        "-B", "200"
    ]

    if no_plots:
        cmd.append("--no-plots")

    if json_output:
        cmd.append("--json-output")
        
    if embedding:
        cmd.append("--track-embedding")
        
    if trace:
        cmd.append("--trace")

    # We assume logs are in utils.torcsLogPath, e.g. /root/.torcs/logs
    cmd.append(utils.torcsLogPath)

    # Capture output
    result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    decoded = result.decode("utf-8", errors="ignore")
    print("Analysis completed.")
    return decoded


def parse_analysis_json(analysis_output):
    """
    Extract the JSON from the '===JSON_START===' / '===JSON_END===' markers
    and parse it into a dictionary.
    """
    start_marker = "===JSON_START==="
    end_marker = "===JSON_END==="
    start_idx = analysis_output.find(start_marker)
    end_idx = analysis_output.find(end_marker, start_idx)
    if start_idx == -1 or end_idx == -1:
        print("Warning: JSON markers not found in analysis output.")
        return {}
    json_str = analysis_output[start_idx + len(start_marker):end_idx].strip()
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {}


def average_metrics(all_results):
    """
    Given a list of dictionaries with numeric values,
    compute the average for each key across all dicts,
    while ignoring NaN or infinite values.
    """
    if not all_results:
        return {}

    # Collect all unique metric keys
    keys = set()
    for d in all_results:
        keys.update(d.keys())

    averages = {}

    for key in keys:
        total = 0.0
        valid_count = 0
        for d in all_results:
            val = d.get(key, None)
            if isinstance(val, (int, float)):
                # Check if val is finite (not NaN or inf)
                if not math.isnan(val) and math.isfinite(val):
                    total += val
                    valid_count += 1
                # else: we skip NaN/inf
        # Only compute average if we had at least one valid numeric value
        if valid_count > 0:
            averages[key] = total / valid_count
        else:
            averages[key] = 0.0 

    return averages

def print_final_results(final_result):
    final_json = json.dumps(final_result, indent=2)
    print("===FINAL_JSON_START===")
    print(final_json)
    print("===FINAL_JSON_END===")
    print("All repetitions done! Exiting now.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-r", "--num_laps", type=int, default=0,
                        help="number of laps to simulate. Overriten by --target-duration if specified.")
    parser.add_argument("--track-export", action="store_true",
                        help="run single-lap track export (with trackexporter bot).")
    parser.add_argument("--json", action="store_true",
                        help="run analysis afterwards and print JSON to stdout.")
    parser.add_argument("--plots", action="store_true", help="enable plots (default: no plots)")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Number of times to run the race+analysis (default=1).")
    parser.add_argument("-d","--target-duration", type=int, 
                        help="Target simulation duration in seconds used to calculate the number of laps. Overrides --num-laps.")
    parser.add_argument("--dont-change-order", action="store_true",
                        help="change the order of bots for each iteration.")
    parser.add_argument("-e","--track-embedding", action="store_true", help="generate track embedding data. This will run a separate simulation to get the embedding data, which will be included in the final JSON output.")
    parser.add_argument("-tr", "--trace", action="store_true", help="return metric trace data for benchmark lap")
    args = parser.parse_args()

    # The usual torcs raceman directory from utils:
    folder_name = utils.torcsRacemanDirectory

    # track export is necessary to get track geometry used in later analyses
    if args.track_export:
        try:
            run_track_export(folder_name)
        except subprocess.CalledProcessError as e:
            print(f"Error running track export: {e}")
            sys.exit(1)
    
    # Only run benchmark if we need to determine num_laps or its specified to get track embeddings
    should_run_benchmark = args.target_duration is not None or args.track_embedding or args.trace
    
    if should_run_benchmark:
        clear_logs() 
        try:
            run_benchmark_sim(folder_name)
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark simulation: {e}")
            sys.exit(1)
        analysis_output = run_analysis(no_plots=True, json_output=True, embedding=args.track_embedding, trace=args.trace)
        parsed = parse_analysis_json(analysis_output)
        
        if args.target_duration is not None:
            lap_times = parsed.get("lap_times", 1)
            lap_time = lap_times['2']
            
            args.num_laps = math.ceil(args.target_duration / lap_time) 
            print(f"Number of laps to simulate was calculated: {args.num_laps}")
        
        if args.track_embedding:
            embedding_data = parsed.get("embedding_data", None)
        
        if args.trace:
            trace_data = {}
            trace_data["speed_trace"] = parsed.get("speed_trace", None)
            trace_data["accel_trace"] = parsed.get("accel_trace", None)
            trace_data["steer_trace"] = parsed.get("steer_trace", None)
            trace_data["brake_trace"] = parsed.get("brake_trace", None)
            
                     
    # 2) run multiple races and accumulate the results
    aggregated_results = []
    for i in range(args.repetitions):
        print(f"\n=== Simulation iteration {i+1}/{args.repetitions} ===")
        try:
            clear_logs() 
            run_race_simulation(folder_name, args.num_laps, i, change_order=not args.dont_change_order)
        except subprocess.CalledProcessError as e:
            print(f"Error running race simulation: {e}")
            sys.exit(1)

        # Possibly do analysis
        if args.json:
            try:
                analysis_output = run_analysis(
                    no_plots=(not args.plots),
                    json_output=True
                )
                # Parse the JSON from the output
                parsed = parse_analysis_json(analysis_output)
                if parsed:
                    aggregated_results.append(parsed)
            except subprocess.CalledProcessError as e:
                print(f"Error running analysis: {e}")
                sys.exit(1)

    final_result = {}
    # 3) After finishing all races+analyses, average the results
    if aggregated_results:
        avg_res = average_metrics(aggregated_results)
        final_result = avg_res
        if args.track_embedding:
            final_result["embedding_data"] = embedding_data
        if args.trace:
            final_result.update(trace_data)
        print_final_results(final_result) 
    else:
        print("Warning: No valid analysis results to average.")
        
    

if __name__ == "__main__":
    main()
