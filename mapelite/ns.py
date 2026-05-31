# Generated from: novelty_search.ipynb
# Converted at: 2026-05-31T09:03:31.065Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import sys
import os
import logging
import structlog

# Suppress DEBUG/INFO in worker processes (which inherit this module but skip the guard).
# setup_logging() inside the guard overrides this for the main process.
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

if __name__ == '__main__':
    # Get the current working directory
    cwd = os.getcwd()
    print(f"Current Working Directory: {cwd}")

    # Define the path to the 'mapelite' folder
    # We assume the notebook is running from the root 'Quality-Diversity-...' folder
    mapelite_path = os.path.join(cwd, 'mapelite')

    # Add it to the system path so Python can find config.py, utils.py, etc.
    if mapelite_path not in sys.path:
        sys.path.append(mapelite_path)
        print(f"Added '{mapelite_path}' to sys.path")

    from mapelite.logging_config import setup_logging
    from mapelite.config import LOG_DIR, LOG_CONSOLE_LEVEL, LOG_FILE_LEVEL
    log_file = setup_logging(
        log_dir=LOG_DIR,
        console_level=LOG_CONSOLE_LEVEL,   # INFO  → shown on console
        file_level=LOG_FILE_LEVEL,         # DEBUG → written to log file
        log_filename=f"novelty_search"
    )
    print(f"Log file: {log_file}")
    from mapelite.logging_config import get_logger
    log = get_logger(__name__)

    import numpy as np
    import random

    from ribs.archives import ProximityArchive
    from ribs.schedulers import Scheduler

    from mapelite.emitter import CustomEmitter
    from mapelite.qd_runner import QDRunner

    from mapelite.config import (
        SOLUTION_DIM,
        BATCH_SIZE,
        NS_DIR,
        BUFFER_FILENAME,
        CHECKPOINT_DIR,
        ELITES_FILENAME,
        HEATMAP_DIR,
        GRIDPLOT_DIR,
        ITERATIONS,
        NS_KNN,
        EMBEDDING_MODEL_PATH,
        PRECOMPILED_EMBEDDINGS_PATH,
        DEFAULT_ARCHIVE_THRESHOLD
    )

    # --- Novelty Search specific config ---
    checkpoint_dir = os.path.join(NS_DIR, CHECKPOINT_DIR)
    heatmap_dir = os.path.join(NS_DIR, HEATMAP_DIR)
    gridplot_dir = os.path.join(NS_DIR, GRIDPLOT_DIR)
    buffer_path = os.path.join(NS_DIR, BUFFER_FILENAME)

    SEED = 67
    ELITES_OUTPUT = os.path.join(NS_DIR, ELITES_FILENAME)
    ALGORITHM_LABEL = "Novelty Search (ProximityArchive + local competition)"

    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)

    # --- Calculate a good novelty threshold from the embedding dataset ---
    # Fits k-NN (same k as the archive) on the pre-existing embeddings and
    # reports percentile distances so you can pick an informed threshold.

    from sklearn.neighbors import NearestNeighbors

    _raw = np.load(PRECOMPILED_EMBEDDINGS_PATH)["embeddings"]

    _k = 15  # same as archive k_neighbors
    _nbrs = NearestNeighbors(n_neighbors=_k + 1).fit(_raw)
    _dists, _ = _nbrs.kneighbors(_raw)
    _knn_mean_per_point = _dists[:, 1:].mean(axis=1)  # exclude self (col 0)

    print(f"Dataset: {len(_raw)} embeddings  |  k={_k}  |  measure dim: {_raw.shape[1]}")
    print(f"  Mean k-NN dist : {_knn_mean_per_point.mean():.4f}")
    for _p in (5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 98.0, 99.0, 99.5, 99.9):
        print(f"  {_p:5.1f}th percentile: {np.percentile(_knn_mean_per_point, _p):.4f}")

    log.info(f"ARCHIVE_THRESHOLD set to {DEFAULT_ARCHIVE_THRESHOLD:.4f}  (manually set in config.py)")

    # --- Initialize directories ---
    os.makedirs(NS_DIR, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(gridplot_dir, exist_ok=True)

    # print cuda device if available
    import torch
    if torch.cuda.is_available():
        log.info(f"CUDA device available: {torch.cuda.get_device_name(0)}")
    else:
        log.info("No CUDA device available, using CPU.")

    # --------------------------------------------------------------
    # Resume from latest checkpoint if available,
    # otherwise build archive from scratch
    # --------------------------------------------------------------
    state = QDRunner.get_state_from_checkpoint(checkpoint_dir)

    _embedding_dim = np.load(PRECOMPILED_EMBEDDINGS_PATH)["embeddings"].shape[1]

    if state["scheduler"] is not None:
        runner = QDRunner.load_state(
            state,
            pretrained_model_path=EMBEDDING_MODEL_PATH,
            checkpoint_dir=checkpoint_dir,
            heatmap_dir=heatmap_dir,
            gridplot_dir=gridplot_dir,
            buffer_path=buffer_path,
            seed=SEED,
        )
    else:
        archive = ProximityArchive(
            solution_dim=SOLUTION_DIM,
            measure_dim=_embedding_dim,
            k_neighbors=NS_KNN,
            novelty_threshold=DEFAULT_ARCHIVE_THRESHOLD,
            seed=SEED,
            local_competition=True
        )
        emitter = CustomEmitter(
            archive,
            solution_dim=SOLUTION_DIM,
            batch_size=BATCH_SIZE,
            bounds=None,
            seed=SEED,
        )

        scheduler = Scheduler(archive, [emitter])

        runner = QDRunner(
            scheduler=scheduler,
            archive=archive,
            pretrained_model_path=EMBEDDING_MODEL_PATH,
            checkpoint_dir=checkpoint_dir,
            heatmap_dir=heatmap_dir,
            gridplot_dir=gridplot_dir,
            buffer_path=buffer_path,
            finetune=True,
            seed=SEED,
        )

    # print ribs version
    import ribs
    print(f"Ribs version: {ribs.__version__}")

    # Run main loop
    global_best_score, global_best_id, stats = runner.run(
        total_iters=ITERATIONS,
        start_iter=state["start_iter"],
    )

    runner.visualizer.plot_stats(title="Novelty Search", stats_dir=NS_DIR)

    runner.visualizer.export_elites(
        output_path=ELITES_OUTPUT,
        algorithm_label=ALGORITHM_LABEL,
        seed=SEED
    )
