import os
import json
import numpy as np
import glob
import time

def create_voronoi_dataset(input_folder, output_filename):
    """
    Reads JSON files from input_folder and saves an NPZ file where:
    - Key: The 'id' from the JSON (converted to string).
    - Value: The 'fitness' dictionary from the JSON.
    """
    
    # Check if folder exists
    if not os.path.exists(input_folder):
        print(f"Error: The directory {input_folder} does not exist.")
        return

    data_store = {}
    file_count = 0

    print("Scanning for files...")
    # Get all json files in the folder
    search_path = os.path.join(input_folder, "*.json")
    files = glob.glob(search_path)
    total_files = len(files)

    print(f"Found {total_files} JSON files. Starting processing...")
    
    start_time = time.time()

    # We use enumerate to keep track of the index (i)
    for i, file_path in enumerate(files, 1):
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
                
                # Extract ID and Fitness
                if 'id' in content and 'fitness' in content:
                    # Npz keys must be strings
                    sample_id = str(content['id']) 
                    fitness_data = content['fitness']
                    
                    data_store[sample_id] = fitness_data
                    file_count += 1
                else:
                    # Print a newline before the error to ensure it doesn't clash with the progress line
                    print(f"\nSkipping {file_path}: Missing 'id' or 'fitness' key.")
                    
        except Exception as e:
            print(f"\nError reading {file_path}: {e}")

        # LOGGING PROGRESS
        # Print status every 100 files or on the very last file
        if i % 100 == 0 or i == total_files:
            percent = (i / total_files) * 100
            # end='\r' overwrites the same line so your terminal doesn't get spammed
            print(f"Processing: {i}/{total_files} ({percent:.1f}%)", end='\r')

    # Print a new line to clear the 'end=\r' from the loop
    print("")

    if file_count > 0:
        print(f"Processing complete. Preparing to save {file_count} entries...")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        # Save compressed npz
        print(f"Saving to '{output_filename}'... (this might take a moment)")
        np.savez_compressed(output_filename, **data_store)
        
        elapsed_time = time.time() - start_time
        print(f"Successfully saved {file_count} entries.")
        print(f"Total time elapsed: {elapsed_time:.2f} seconds.")
    else:
        print("No valid data found to save.")

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = os.path.join("data","Archive", "voronoi tita winded", "fitted")
    OUTPUT_FILE = os.path.join("qd", "embeddings", "datasets", "fitness_dict_mixedRng_tita_winded.npz")

    create_voronoi_dataset(INPUT_DIR, OUTPUT_FILE)