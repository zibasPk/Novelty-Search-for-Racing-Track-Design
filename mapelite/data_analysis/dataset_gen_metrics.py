import json
import numpy as np
from pathlib import Path
import sys
import os

cwd = os.getcwd()
print(f"Current Working Directory: {cwd}")

# Define the path to the 'mapelite' folder
# We assume the notebook is running from the root 'Quality-Diversity-...' folder
mapelite_path = os.path.join(cwd, 'mapelite')

# Add it to the system path so Python can find config.py, utils.py, etc.
if mapelite_path not in sys.path:
    sys.path.append(mapelite_path)
    print(f"Added '{mapelite_path}' to sys.path")
    
import utils


def create_dataset(source_folder, output_file, max_files=None):
    dataset = []
    sourceIDs = []
    max_length = 0
    
    source_path = Path(source_folder)
    if not source_path.exists():
        print(f"Error: Folder '{source_folder}' not found.")
        return

    json_files = list(source_path.glob("*.json"))
    print(f"Found {len(json_files)} files.")

    invalid_fitness_count = 0
    no_embedding_count = 0

    for index, file_path in enumerate(json_files):
        if max_files is not None and index >= max_files:
            break
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                fitness = data.get('fitness')
                if not isinstance(fitness, dict) or not fitness:
                    invalid_fitness_count += 1
                    print(
                        f"Warning: Invalid or missing fitness data in {file_path.name}, skipping this file.")
                    continue
                metrics = fitness.get('embedding_data', [])

                if not isinstance(metrics, (list, np.ndarray)) or len(metrics) == 0:
                    no_embedding_count += 1
                    print(f"Warning: No 'embedding_data' in {file_path.name}, skipping this file.")
                    continue
                
                if len(metrics) > max_length:
                    max_length = len(metrics)
                    
                dataset.append(np.array(metrics, dtype=np.float32))
                sourceIDs.append(data.get("id"))
                         
        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

    if not dataset:
        print("No data collected. Exiting.")
        return

    # Pad sequences to the same length
    data_to_save = []
    masks = []
    for metrics in dataset:
        length = len(metrics)
        if length < max_length:
            pad_width = ((0, max_length - length), (0, 0))
            padded_metrics = np.pad(metrics, pad_width, mode='constant', constant_values=0)
            mask = np.concatenate([np.ones(length, dtype=bool), np.zeros(max_length - length, dtype=bool)])
        else:
            padded_metrics = metrics
            mask = np.ones(max_length, dtype=bool)
        data_to_save.append(padded_metrics)
        masks.append(mask)
    
    
    np.savez_compressed(output_file, data=data_to_save, masks=masks, ids=sourceIDs)

    print(f"Success! Created a dataset with {len(dataset)} flattened rows.")
    print(f"Max embedding length: {max_length}")
    print(f"Files without 'embedding_data': {no_embedding_count}")
    print(f"Files with invalid fitness data: {invalid_fitness_count}")

if __name__ == "__main__":
    SOURCE_DIR = 'data/voronoi/fitted'
    # Changed extension to .npz
    OUTPUT_NAME = 'data/datasets/metrics/dataset4k.npz'

    # Ensure output directory exists
    Path('data/datasets/metrics').mkdir(parents=True, exist_ok=True)

    create_dataset(SOURCE_DIR, OUTPUT_NAME, 4000)
