# utilities.py

import numpy as np
import requests
import random
import joblib
import os

from config import (
    BASE_URL, GENERATION_MODE, POINTS_COUNT, MAX_SELECTED_CELLS, SOLUTION_DIM,
    TRACK_SIZE_RANGE, INVALID_SCORE, CHECKPOINT_DIR
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


# --- Global Stats for Normalization ---
_STATS = {
    'len_min': np.inf, 'len_max': -np.inf,
    'ov_min': np.inf, 'ov_max': -np.inf,
    'dx_min': np.inf, 'dx_max': -np.inf,
    'bend_min': np.inf, 'bend_max': -np.inf,
}

def _upd(k, v):
    """Update min/max for a given statistic key."""
    lo, hi = f'{k}_min', f'{k}_max'
    if v < _STATS[lo]: _STATS[lo] = v
    if v > _STATS[hi]: _STATS[hi] = v

def _norm(k, v, eps=1e-6):
    """Normalize a value based on the current global min/max stats."""
    lo, hi = _STATS[f'{k}_min'], _STATS[f'{k}_max']
    range_val = hi - lo
    if range_val == 0.0:
        return 0.0 # Avoid division by zero if all values are the same
    return (v - lo) / (range_val + eps)

# --- Core Utility Functions ---

def generate_solution(iteration):
    """Generates a new track solution by calling the external API."""
    # print(f"Generating solution for iteration {iteration}") # Mute: too chatty
    try:
        response = requests.post(
            f"{BASE_URL}/generate",
            json={
                "id": iteration + random.random(),
                "mode": GENERATION_MODE,
                "trackSize": random.randint(TRACK_SIZE_RANGE[0], TRACK_SIZE_RANGE[1])
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error generating solution for iteration {iteration}: {e}")
        return None

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
            
    # 3. Solution ID (last element)
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
    for i in range(POINTS_COUNT * 2, SOLUTION_DIM - 1, 2):
        x_val = arr[i]
        y_val = arr[i+1]
        # Assuming (0, 0) is a sentinel value for unused slots
        if x_val != 0 or y_val != 0:
            sel.append({"x": float(x_val), "y": float(y_val)})
            
    return {
        "id": float(arr[-1]),
        "mode": GENERATION_MODE,
        "dataSet": ds,
        "selectedCells": sel
    }

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

def descriptor_from_track(sol):
    """Converts the track's spline vector into the 2D behavioral descriptor using UMAP."""
    # The 'splineVector' is assumed to be part of the evaluation JSON response
    pts = np.array([[p["x"], p["y"]] for p in sol.get("splineVector", [])], dtype=float)
    
    # Align the spline to account for rotation/translation invariance
    aligned = pca_align(pts)
    
    # Flatten the aligned points to create the feature vector
    flat = aligned.ravel()
    
    # Transform the feature vector using the pre-trained UMAP model
    # Note: flat[None, :] ensures the input has the correct shape (1, N)
    return EMBEDDING_MODEL.transform(flat[None, :])[0]

def fitness_formula(fit):
    """Calculates the scalar fitness score based on evaluation metrics."""
    length = max(fit.get('length', 0.0), 1e-3)
    bend_len = fit.get('right_len', 0.0) + fit.get('left_len', 0.0)
    overtakes = fit.get('total_overtakes', 0.0)
    dx = abs(fit.get('deltaX', 0.0)) or 1e-3

    bend_ratio = bend_len / length

    # Update global statistics for normalization
    for k, v in (('len', length), ('ov', overtakes), ('dx', dx),
                 ('bend', bend_ratio)):
        _upd(k, v)

    # Calculate normalized score
    score = (
        0.25 * _norm('len',  length) +  # encourage longer tracks
        0.60 * _norm('bend', bend_ratio) +  # maximise curves per metre
        0.15 * (_norm('ov',  overtakes) /
                (_norm('dx', dx) + 1e-3))  # overtakes, damped by Δx
    )
    return float(score)

def evaluate_solution(sol):
    """Submits a solution to the external API for evaluation and computes descriptor/fitness."""
    sol_id = sol.get("id", 0)
    ok = True
    msg = ""
    desc = np.zeros((2,))  # Default descriptor
    fit_score = INVALID_SCORE

    try:
        # 1. Send solution for evaluation
        r = requests.post(f"{BASE_URL}/evaluate", json=sol, timeout=60)
        r.raise_for_status()
        r_json = r.json()
        
        # 2. Extract raw fitness metrics and compute descriptor
        fit = r_json.get("fitness", {})
        desc = descriptor_from_track(r_json)  
        
        # 3. Compute final fitness score (will update global _STATS here)
        fit_score = fitness_formula(fit)
        
    except Exception as e:
        ok = False
        msg = str(e)
        
    return sol_id, ok, msg, fit_score, desc


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