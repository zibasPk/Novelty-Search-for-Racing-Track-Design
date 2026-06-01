# config.py

import logging
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
DEFAULT_START_ITER = 0
ITERATIONS = 1000
RANDOM_POPULATION_ITERS = 10
BATCH_SIZE = 20
INVALID_SCORE = -1e9

RETRAIN_EVERY = 15
MEASURE_DIM = 32


# --- Archive Parameters ---
NS_KNN = 15  # for kNN novelty calculation
DEFAULT_ARCHIVE_THRESHOLD = 6.3
RECALC_THRESHOLD_EVERY = 10
TARGET_ARCHIVE_SIZE = 500
K_CSC = 1e-5


# --- Checkpointing---
CHECKPOINT_EVERY = 50

# --- Directories ---
EMBEDDING_MODEL_PATH = "mapelite/embeddings/models/model_metrics_VAE/model_metrics_VAE_mixRng_tita_circular_7.pth"
PRECOMPILED_EMBEDDINGS_PATH = "mapelite/datasets/track_embeddings_metrics_32dim_rngMixDS_tita_circular_7.npz"
NS_DIR = "data/ns/"
CHECKPOINT_DIR = "checkpoints/"
HEATMAP_DIR = "heatmaps/"
GRIDPLOT_DIR = "gridplots/"
STATS_FILENAME = "stats.pkl"
BUFFER_FILENAME = "buffer.json"
ELITES_FILENAME = "elites.json"

# --- Logging ---
LOG_DIR = "logs"
LOG_CONSOLE_LEVEL = logging.INFO
LOG_FILE_LEVEL = logging.DEBUG

BUFFER_DIR = "buffers/"

