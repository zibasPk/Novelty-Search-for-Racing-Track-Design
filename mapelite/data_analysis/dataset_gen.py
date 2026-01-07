import json
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
import utils


def create_flattened_dataset(source_folder, output_file, max_files=None):
    dataset = []
    sourceIDs = []
    source_path = Path(source_folder)

    if not source_path.exists():
        print(f"Error: Folder '{source_folder}' not found.")
        return

    json_files = list(source_path.glob("*.json"))
    print(f"Found {len(json_files)} files. Flattening...")

    no_spline_count = 0
    nan_spline_count = 0
    invalid_fitness_count = 0

    for index, file_path in enumerate(json_files):
        if max_files is not None and index >= max_files:
            break
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                spline = data.get("splineVector", [])
                if isinstance(spline, (list, np.ndarray)) and len(spline) > 0:
                    
                    # check if fitness data is present
                    if not isinstance(data.get('fitness'), dict) or not data['fitness']:
                        invalid_fitness_count += 1
                        print(
                            f"Warning: Invalid or missing fitness data in {file_path.name}, skipping this file.")
                        continue
                      
                    # Align using PCA
                    points = np.array([[p["x"], p["y"]] for p in data.get("splineVector", [])], dtype=float)
                    aligned = utils.pca_align(points)
                    # Flatten into 1D vector
                    flat_vector = aligned.ravel()
                    
                    
                    nan_count = np.isnan(flat_vector).sum()
                    if nan_count > 0:
                        nan_spline_count += 1
                        print(
                            f"Warning: Found {nan_count} NaN values in {file_path.name}, skipping this file.")
                        continue
                    
                    sourceIDs.append(data.get("id"))
                    dataset.append(flat_vector)
                else:
                    no_spline_count += 1
                    print(f"Warning: No 'splineVector' in {file_path.name}")

        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

    if not dataset:
        print("No data collected. Exiting.")
        return

    # Check if all vectors are the same length
    lengths = [len(v) for v in dataset]
    all_same_length = all(l == lengths[0] for l in lengths)

    if all_same_length:
        # Convert to a standard 2D matrix (Samples, Features)
        data_to_save = np.array(dataset, dtype=np.float32)
        print(f"Matrix shape: {data_to_save.shape}")
    else:
        # If lengths vary, save as an object array
        print("Warning: Spline vectors have varying lengths. Saving as object array.")
        data_to_save = np.array(dataset, dtype=object)

    # Save as compressed npz
   
    np.savez_compressed(output_file, splines=data_to_save, ids=np.array(sourceIDs))

    print(f"Success! Created a dataset with {len(dataset)} flattened rows.")
    print(f"Files without 'splineVector': {no_spline_count}")
    print(f"Files with NaN in 'splineVector': {nan_spline_count}")
    print(f"Files with invalid/missing fitness data: {invalid_fitness_count}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    SOURCE_DIR = 'data/voronoi/fitted'
    # Changed extension to .npz
    OUTPUT_NAME = 'data/dataset10k.npz'

    # Ensure output directory exists
    Path('data').mkdir(parents=True, exist_ok=True)

    create_flattened_dataset(SOURCE_DIR, OUTPUT_NAME, 10000)
