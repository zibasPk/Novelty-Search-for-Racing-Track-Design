import numpy as np

# 1. Load the data
print("--- LOADING DATA ---")
# If your latents/ids are in the same file or different files, adjust accordingly.
# I will assume 'tracks.npz' is what you tried to load as the dictionary.
try:
    tracks_data = np.load("datasets/tracks.npz", allow_pickle=True)
    print("✅ File loaded successfully.")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# 2. Inspect the Keys
keys = list(tracks_data.keys())
print(f"\n--- GENERAL STATS ---")
print(f"Total number of items (keys) in file: {len(keys)}")
print(f"First 5 keys: {keys[:5]}")

# 3. Inspect a Single Track
if len(keys) > 0:
    first_key = keys[0]
    first_track = tracks_data[first_key]
    
    print(f"\n--- SAMPLE TRACK ({first_key}) ---")
    print(f"Type: {type(first_track)}")
    
    if isinstance(first_track, np.ndarray):
        print(f"Shape: {first_track.shape}")
        print(f"Data Type: {first_track.dtype}")
        # Check for NaNs or Infinite values which crash WebGL
        print(f"Contains NaN? {np.isnan(first_track).any()}")
        print(f"Contains Inf? {np.isinf(first_track).any()}")
    else:
        print("Wait, this is not a numpy array. It is:", first_track)

# 4. Latents Check (Critical for WebGL crash)
# Are you loading latents from here too? Or a separate list?
# If you have your latents variable ready, print its shape:
# print(f"Latents Shape: {latents.shape}")