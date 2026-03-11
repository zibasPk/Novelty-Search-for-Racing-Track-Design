# utilities.py

import numpy as np
import requests
import random
import joblib
import os

from config import (
    BASE_URL, GENERATION_MODE, POINTS_COUNT, MAX_SELECTED_CELLS, SOLUTION_DIM, INVALID_SCORE, RngMode
)

# --- Embedding Model Loading ---
# Note: Ensure "data/EmbeddingModels/umap_model.joblib" exists relative to where the notebook is run.
try:
    EMBEDDING_MODEL = joblib.load("data/EmbeddingModels/umap_model.joblib")
except FileNotFoundError:
    print("Warning: UMAP model not found. Placeholder used. (Run the setup notebook first?)")
    # Placeholder class to prevent crash if model isn't trained/found
    class PlaceholderUMAP:
        def transform(self, data):
            # Returns a dummy 2D descriptor
            return np.zeros((data.shape[0], 2))
    EMBEDDING_MODEL = PlaceholderUMAP()

# --- Core Utility Functions ---

def solution_to_array(sol):
    """Converts a JSON solution dictionary into the flat NumPy array format required by MAP-Elites."""
    if sol is None:
        return None
    arr = np.zeros(SOLUTION_DIM)
    
    # 1. Data Points (x, y)
    for i, p in enumerate(sol.get("dataSet", [])):
        arr[i * 2] = p.get("x", 0)
        arr[i * 2 + 1] = p.get("y", 0)
        
    # 2. Selected Cells (x, y) - capped at MAX_SELECTED_CELLS
    for i, c in enumerate(sol.get("selectedCells", [])):
        if i < MAX_SELECTED_CELLS:
            idx = POINTS_COUNT * 2 + i * 2
            arr[idx] = c.get("x", 0)
            arr[idx + 1] = c.get("y", 0)

    # 3. rngMode (second-to-last element): 0 = uniform, 1 = perlin
    rng_mode = sol.get("rngMode", RngMode.UNIFORM)
    arr[-2] = rng_mode

    # 4. Solution ID (last element)
    arr[-1] = sol.get("id", 0)
    return arr

def array_to_solution(arr):
    """Converts the flat NumPy array back into the JSON solution dictionary format."""
    ds = []
    # 1. Data Points
    for i in range(0, POINTS_COUNT * 2, 2):
        ds.append({"x": float(arr[i]), "y": float(arr[i+1])})
        
    sel = []
    # 2. Selected Cells (only include non-zero/valid cells)
    sel_end = POINTS_COUNT * 2 + MAX_SELECTED_CELLS * 2
    for i in range(POINTS_COUNT * 2, sel_end, 2):
        x_val = arr[i]
        y_val = arr[i+1]
        # Assuming (0, 0) is a sentinel value for unused slots
        if x_val != 0 or y_val != 0:
            sel.append({"x": float(x_val), "y": float(y_val)})

    # 3. rngMode (second-to-last element): 0 = uniform, 1 = perlin
    rng_val = int(arr[-2])
    try:
        rng_mode = RngMode(rng_val)
    except ValueError:
        rng_mode = RngMode.UNIFORM  # fallback for invalid/sentinel (e.g. INVALID_SCORE fill)

    return {
        "id": float(arr[-1]),
        "mode": GENERATION_MODE,
        "rngMode": rng_mode,
        "dataSet": ds,
        "selectedCells": sel
    }
    
def invalid_solution_array(id = INVALID_SCORE):
    """Returns a flat array filled with INVALID_SCOREs except for the last element which holds the ID, used to represent invalid solutions."""
    arr = np.full(SOLUTION_DIM, INVALID_SCORE)
    arr[-1] = id
    return arr

def is_valid_solution_array(arr):
    """Checks if a solution array is valid (not filled with INVALID_SCORE)."""
    return arr is not None and not np.all(arr[:-1] == INVALID_SCORE)  # Ignore ID in validity check

def get_fractional_part(x):
    """Gets the fractional part of a float ID, used for mutation/crossover."""
    return x - int(x)

def pca_align(points):
    """Performs PCA-based alignment on a set of 2D points (spline vector)."""
    # Center the points
    pts = points - points.mean(0)
    
    # Perform SVD (equivalent to PCA for the first two components)
    u, _, _ = np.linalg.svd(pts, full_matrices=False)
    
    # Calculate rotation angle to align the principal component with the x-axis
    angle = np.arctan2(u[1, 0], u[0, 0])
    rot = np.array([[np.cos(-angle), -np.sin(-angle)],
                    [np.sin(-angle),  np.cos(-angle)]])
    
    # Apply rotation
    aligned = pts @ rot.T
    
    # Ensure the alignment direction is consistent (e.g., first point x-coordinate > 0)
    if aligned[0, 0] < 0:
        aligned[:, 0] *= -1
        
    return aligned



def get_initial_archive_ranges(umap_model = EMBEDDING_MODEL, padding=0.1, default_bounds=(-1, 1)):
    """
    Extracts the bounds of the offline training data from the UMAP model to 
    set intelligent initial ranges for the MAP-Elites grid.
    
    Args:
        umap_model: The loaded UMAP object.
        padding: Percentage of extra space to add around the data (0.1 = 10%).
        default_bounds: Fallback range if training data is missing from the model object.
    """
    # Check if the model contains the training embedding
    if hasattr(umap_model, 'embedding_') and umap_model.embedding_ is not None:
        data = umap_model.embedding_
        
        # Calculate X bounds
        min_x, max_x = data[:, 0].min(), data[:, 0].max()
        span_x = max_x - min_x
        range_x = (min_x - (span_x * padding), max_x + (span_x * padding))
        
        # Calculate Y bounds
        min_y, max_y = data[:, 1].min(), data[:, 1].max()
        span_y = max_y - min_y
        range_y = (min_y - (span_y * padding), max_y + (span_y * padding))
        
        print(f"[Archive Init] Auto-detected ranges: X={range_x}, Y={range_y}")
        return [range_x, range_y]
    
    else:
        # Fallback if the model was saved without data to save space
        print(f"[Archive Init] Model data missing. Using default ranges: {default_bounds}")
        return [default_bounds, default_bounds]