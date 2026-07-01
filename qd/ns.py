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

    # Define the path to the 'qd' folder
    # We assume the notebook is running from the root 'Quality-Diversity-...' folder
    qd_path = os.path.join(cwd, 'qd')

    # Add it to the system path so Python can find config.py, utils.py, etc.
    if qd_path not in sys.path:
        sys.path.append(qd_path)
        print(f"Added '{qd_path}' to sys.path")

    from qd.logging_config import setup_logging
    from qd.config import LOG_DIR, LOG_CONSOLE_LEVEL, LOG_FILE_LEVEL
    log_file = setup_logging(
        log_dir=LOG_DIR,
        console_level=LOG_CONSOLE_LEVEL,   # INFO  → shown on console
        file_level=LOG_FILE_LEVEL,         # DEBUG → written to log file
        log_filename=f"novelty_search"
    )
    print(f"Log file: {log_file}")
    from qd.logging_config import get_logger
    log = get_logger(__name__)

    import numpy as np
    import random

    from ribs.archives import ProximityArchive
    from ribs.schedulers import Scheduler

    from qd.emitter import CustomEmitter
    from qd.qd_runner import QDRunner

    from qd.config import (
        SOLUTION_DIM,
        BATCH_SIZE,
        NS_DIR,
        BUFFER_FILENAME,
        CHECKPOINT_DIR,
        ELITES_FILENAME,
        ITERATIONS,
        NS_KNN,
        EMBEDDING_MODEL_PATH,
        PRECOMPILED_EMBEDDINGS_PATH,
        DEFAULT_ARCHIVE_THRESHOLD,
        DO_FINETUNE,
        MEASURE_DIM
    )

    # --- Novelty Search specific config ---
    checkpoint_dir = os.path.join(NS_DIR, CHECKPOINT_DIR)
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

    log.info(f"ARCHIVE_THRESHOLD set to {DEFAULT_ARCHIVE_THRESHOLD:.4f}  (manually set in config.py)")

    # --- Initialize directories ---
    os.makedirs(NS_DIR, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

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

    _embedding_dim = MEASURE_DIM

    if state["scheduler"] is not None:
        runner = QDRunner.load_state(
            state,
            pretrained_model_path=EMBEDDING_MODEL_PATH,
            checkpoint_dir=checkpoint_dir,
            buffer_path=buffer_path,
            seed=SEED,
            do_retraining=DO_FINETUNE,
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
            buffer_path=buffer_path,
            finetune=DO_FINETUNE,
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
    runner.visualizer.plot_finetuning_val_loss(stats_dir=NS_DIR)
    runner.visualizer.plot_finetuning_val_kld(stats_dir=NS_DIR)

    runner.visualizer.export_elites(
        output_path=ELITES_OUTPUT,
        algorithm_label=ALGORITHM_LABEL,
        seed=SEED
    )
