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
    


def create_dataset(source_folder, output_file, max_files=None):
    dataset = []
    sourceIDs = []
    lengths = []
  
    
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
                
                lengths.append(len(metrics))
               
                dataset.append(np.array(metrics, dtype=np.float32))
                sourceIDs.append(data.get("id"))
                         
        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

    if not dataset:
        print("No data collected. Exiting.")
        return

    # flatten the dataset and create an index array
    flattened_dataset = np.concatenate(dataset, axis=0)
    lengths = np.array([len(track) for track in dataset])
    index_array = np.cumsum(lengths[:-1])
        
    max_length = np.max(lengths)
    min_length = np.min(lengths)
    avg_length = np.mean(lengths)
    std_dev_length = np.std(lengths)
    
    np.savez_compressed(output_file, data=flattened_dataset, indices=index_array, ids=sourceIDs)

    print(f"Success! Created a dataset with {len(dataset)} flattened rows.")
    print(f"Max embedding length: {max_length}")
    print(f"Min embedding length: {min_length}")
    print(f"Average embedding length: {avg_length}")
    print(f"Standard deviation of embedding length: {std_dev_length}")
    print(f"Files without 'embedding_data': {no_embedding_count}")
    print(f"Files with invalid fitness data: {invalid_fitness_count}")

if __name__ == "__main__":
    SOURCE_DIR = 'data/voronoi tita winded/fitted'
    # Changed extension to .npz
    OUTPUT_DIR = 'mapelite/embeddings/datasets/'
    OUTPUT_NAME = 'dataset20k_mixedRng_tita_winded.npz'

    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    create_dataset(SOURCE_DIR, os.path.join(OUTPUT_DIR, OUTPUT_NAME), 20000)
