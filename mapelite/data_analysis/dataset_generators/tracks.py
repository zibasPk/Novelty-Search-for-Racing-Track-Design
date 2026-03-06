import json
import numpy as np

INPUT_FILE = 'tracks_mixedRng.json'
OUTPUT_FILE = 'tracks_mixedRng.npz'

def json_to_npz(input_file, output_file):
    print(f"Loading {input_file}...")
    
    # Load the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} tracks")
    
    # Extract seeds
    seeds = [item['seed'] for item in data]
    
    # Check track lengths
    track_lengths = [len(item['track']) for item in data]
    print(f"Track lengths - min: {min(track_lengths)}, max: {max(track_lengths)}")
    
    # Create a dictionary with seed numbers as keys
    save_dict = {}
    for item in data:
        save_dict[str(item['seed'])] = np.array(item['track'], dtype=np.float32)
    
    np.savez_compressed(output_file, **save_dict)
    
    print(f"Successfully saved {len(save_dict)} tracks to {output_file}")

if __name__ == '__main__':
    json_to_npz(INPUT_FILE, OUTPUT_FILE)