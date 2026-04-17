# utilities.py

import numpy as np

from mapelite.config import (
    GENERATION_MODE, POINTS_COUNT, MAX_SELECTED_CELLS, SOLUTION_DIM, INVALID_SCORE, RngMode
)

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
