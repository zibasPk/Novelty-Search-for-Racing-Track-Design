# config.py

import os
from enum import IntEnum


class RngMode(IntEnum):
    UNIFORM = 0
    PERLIN = 1


# --- General Setup ---
BASE_URL = 'http://localhost:4242'
GENERATION_MODE = 'voronoi'  # 'convexHull' or 'voronoi'

# --- Track/Solution Parameters ---
POINTS_COUNT = 100
MAX_SELECTED_CELLS = 10  # relevant only for voronoi
# Calculated Dimension: POINTS_COUNT * 2 (x/y) + MAX_SELECTED_CELLS * 2 (x/y) + 1 (rngMode) + 1 (ID)
# All cells coordinates + selected cell coordinates + rngMode + solution ID
SOLUTION_DIM = POINTS_COUNT * 2 + MAX_SELECTED_CELLS * 2 + 1 + 1
TRACK_SIZE_RANGE = (4, 10)  # (4, 10) for voronoi otherwise (100, 100)

# --- QD Parameters ---
ITERATIONS = 1000
RANDOM_POPULATION_ITERS = 100
BATCH_SIZE = 10
INVALID_SCORE = -1e9

# --- Archive Parameters ---
ARCHIVE_BINS = 30  # cells per axis
REMAPPING_EVERY = 100  # remap archive every N iterations
BUFFER_SIZE = 1000  # keep last 1000 solutions

# --- Checkpointing and Debugging ---
CHECKPOINT_EVERY = 50
DEBUG_CROSSOVER = True
DEBUG_MUTATION = True

# --- Directories (Create if not exist in main notebook setup) ---
CHECKPOINT_DIR = "data/checkpoints/"
HEATMAP_DIR = "data/heatmaps/"
STATS_DIR = "data/stats/"
BUFFER_DIR = "data/buffers/"

