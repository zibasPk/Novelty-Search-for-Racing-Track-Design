# qd_runner.py
# Shared infrastructure for Quality-Diversity search loops.
# novelty_search.ipynb delegates to these classes.

import dask
import torch

from mapelite.utils import array_to_solution
from mapelite.config import (
    BATCH_SIZE,
    BUFFER_DIR,
    CHECKPOINT_EVERY,
    INVALID_SCORE,
    DEFAULT_START_ITER,
    RETRAIN_EVERY,
    NS_KNN,
    MEASURE_DIM,
    DEFAULT_ARCHIVE_THRESHOLD,
    RECALC_THRESHOLD_EVERY,
    RANDOM_POPULATION_ITERS,
    TARGET_ARCHIVE_SIZE,
    K_CSC
)
from mapelite.evaluator import EvaluatorMetrics
from mapelite.archive_visualizer import ArchiveVisualizer
from mapelite.vae import MetricsVAE, MetricsPreprocessor, MetricsDataset, TrainingConfig, VAETrainer
from mapelite.vae.data import collate_fn
from mapelite.vae.config import FINETUNING_CONFIG as _FT, DATA_CONFIG as _DC
from torch.utils.data import DataLoader, random_split
from mapelite.logging_config import get_logger
from mapelite import qd_stats

from dask.distributed import Client, LocalCluster
import numpy as np
from numpy_groupies import aggregate_nb as aggregate
import os
import glob
import pickle
import datetime
import json
import joblib
import copy
from contextlib import contextmanager
from sklearn.neighbors import NearestNeighbors

log = get_logger(__name__)


# ── Evaluation Buffer ───────────────────────────────────────────────────────

class EvaluationBuffer:
    """Accumulates every evaluated track (id, spline data, embedding) and
    persists them to a JSON file in ``data/buffers/``.

    Entries are keyed by sol_id, so each id is stored only once
    (last-write-wins on duplicate ids).  It is saved at every checkpoint
    interval and once more when the loop finishes.
    """

    def __init__(self, buffer_path: str):
        self.buffer_path = buffer_path
        os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
        self.entries: dict = {}
        self._load()

    # -- persistence ----------------------------------------------------------

    def _load(self):
        """Resume from an existing buffer file, if present."""
        if os.path.exists(self.buffer_path):
            with open(self.buffer_path, "r") as f:
                data = json.load(f)
            self.entries = {e["id"]: e for e in data.get("tracks", [])}
            log.info("Buffer resumed", count=len(
                self.entries), path=self.buffer_path)
        else:
            log.info("Buffer empty — starting fresh", path=self.buffer_path)

    def save(self):
        """Write the full buffer to disk."""
        payload = {
            "total": len(self.entries),
            "timestamp": datetime.datetime.now().isoformat(),
            "tracks": list(self.entries.values()),
        }
        with open(self.buffer_path, "w") as f:
            json.dump(payload, f)
        log.info("Buffer saved", count=len(
            self.entries), path=self.buffer_path)

    def get_phenotype_data(self, sol_ids):
        """Return phenotype data for a list of solution IDs, if available."""
        data = []
        for sol_id in sol_ids:
            entry = self.entries.get(sol_id)
            if entry and "phenotype_data" in entry:
                data.append(entry["phenotype_data"])
            else:
                data.append(None)
        return data

    def clear(self):
        """Clear the buffer and delete the file on disk."""
        self.entries = {}
        if os.path.exists(self.buffer_path):
            os.remove(self.buffer_path)
            log.info("Buffer cleared and file deleted", path=self.buffer_path)
        else:
            log.info("Buffer cleared (no file to delete)", path=self.buffer_path)

    def record(self, sol_id, sol_dict: dict, measure, score: float, ok: bool, phenotype_data=None):
        """Record a single evaluated track.

        Parameters
        ----------
        sol_id : float | int
            Unique solution identifier.
        sol_dict : dict
            Solution dictionary produced by ``array_to_solution`` containing
            ``dataSet`` and ``selectedCells`` (the data needed to reconstruct
            the spline).
        measure : array-like
            Behavioural descriptor (measure) / embedding vector.
        score : float
            Fitness score returned by the evaluator.
        ok : bool
            Whether the evaluation succeeded.
        """
        self.entries[sol_id] = {
            "id": sol_id,
            "dataSet": sol_dict.get("dataSet", []),
            "selectedCells": sol_dict.get("selectedCells", []),
            "mode": sol_dict.get("mode", ""),
            "rngMode": sol_dict.get("rngMode", "uniform"),
            "measure": np.asarray(measure).tolist(),
            "score": float(score),
            "valid": ok,
            "phenotype_data": phenotype_data,
        }

    def __getitem__(self, sol_id):
        return self.entries[sol_id]

    def __contains__(self, sol_id):
        return sol_id in self.entries

    def __len__(self):
        return len(self.entries)


# ── QD Runner ───────────────────────────────────────────────────────────────

def _eval_on_worker(evaluator, sol):
    """Thin wrapper so Dask receives the evaluator as an already-scattered future."""
    return evaluator.evaluate(sol)


class QDRunner:
    """Encapsulates the ask → evaluate → tell QD loop and its associated state.

    Parameters
    ----------
    scheduler : pyribs scheduler
        The QD scheduler (wraps archive + emitters).
    archive : pyribs archive
        The QD archive (also accessible via ``scheduler.archive``).
    client : dask.distributed.Client
        Dask distributed client for parallel evaluation.
    evaluator_future : dask Future
        Evaluator scattered to all Dask workers.
    checkpoint_dir, heatmap_dir, gridplot_dir : str
        Output directories for checkpoints and plots.
    stats_path : str
        Path to the pickled stats file.
    buffer_path : str, optional
        Path to the evaluation buffer JSON.  Defaults to ``BUFFER_DIR/buffer.json``.
    seed : int, optional
        Random seed (used by UMAP visualizations).
    """

    def __init__(
        self,
        scheduler,
        archive,
        checkpoint_dir,
        heatmap_dir,
        gridplot_dir,
        pretrained_model_path=None,
        finetune=False,
        start_iter=DEFAULT_START_ITER,
        buffer_path=None,
        seed=None,
        stats=None,
        grid_state=None
    ):
        self.scheduler = scheduler
        self.archive = archive

        self.stats: list[dict] = stats if stats is not None else []

        self.do_finetune = finetune
        self.start_iter = start_iter

        self.checkpoint_dir = checkpoint_dir
        self.heatmap_dir = heatmap_dir
        self.gridplot_dir = gridplot_dir
        self.buffer_path = buffer_path or os.path.join(
            BUFFER_DIR, "buffer.json")

        self.seed = seed
        # Mutable run state
        self.global_best_score = stats[-1].get("global_best_score", INVALID_SCORE) if stats else INVALID_SCORE
        self.global_best_id = stats[-1].get("global_best_id") if stats else None
        self.iteration = 0

        # Evaluation buffer & visualizer
        self._evaluation_buffer = EvaluationBuffer(self.buffer_path)
        self._visualizer = ArchiveVisualizer(
            archive, self.stats, heatmap_dir, gridplot_dir, seed=seed, grid_state=grid_state)

        self._embedding_model = None
        self.use_finetuning = finetune
        if finetune:
            log.info("Retraining enabled: will finetune evaluator on elites every "
                     f"{RETRAIN_EVERY} iterations and recalculate measures for all archived elites.")
            if pretrained_model_path is not None:
                client, cluster, evaluator_future, self._embedding_model = self.setup_dask(batch_size=BATCH_SIZE, model_path=pretrained_model_path)
            else:
                log.info("No pretrained model path provided for retraining mode; initializing new model and scattering to Dask workers.")
                self.use_finetuning = False
                self._embedding_model = MetricsVAE()
                client, cluster, evaluator_future, _ = self.setup_dask(batch_size=BATCH_SIZE, model=self._embedding_model, embedding_dim=MEASURE_DIM)
        else:
            log.info("Retraining disabled: using fixed evaluator for entire run.")
            client, cluster, evaluator_future, self._embedding_model = self.setup_dask(batch_size=BATCH_SIZE, model_path=pretrained_model_path)

        self.client = client
        self.cluster = cluster
        self.evaluator_future = evaluator_future

    @classmethod
    def load_state(cls, state, pretrained_model_path, checkpoint_dir, heatmap_dir, gridplot_dir, buffer_path, seed, do_retraining=False):
        instance = cls(
            scheduler=state["scheduler"],
            archive=state["archive"],
            start_iter=state["start_iter"],
            stats=state["stats"],
            pretrained_model_path=pretrained_model_path,
            checkpoint_dir=checkpoint_dir,
            heatmap_dir=heatmap_dir,
            gridplot_dir=gridplot_dir,
            buffer_path=buffer_path,
            seed=seed,
            grid_state=state["stats"][-1].get("grid_state", None),
            do_retraining= do_retraining
        )
        return instance

    @staticmethod
    def get_state_from_checkpoint(checkpoint_dir):
        """Check for existing checkpoints and return the latest state if found."""

        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pkl")))

        scheduler = None
        archive = None
        start_iter = DEFAULT_START_ITER
        stats = None
        seed = None

        if not checkpoints:
            log.info("No checkpoint found — starting fresh")
            return {
                "scheduler": scheduler,
                "archive": archive,
                "start_iter": start_iter,
                "stats": stats,
                "seed": None
            }

        latest_ckpt = checkpoints[-1]
        with open(latest_ckpt, "rb") as f:
            state = pickle.load(f)

        scheduler = state["scheduler"]
        archive = scheduler.archive
        # Resume from the next iteration after the checkpointed one
        start_iter = state["iteration"] + 1
        scheduler.emitters[0].iteration = start_iter


        log.info("Checkpoint loaded", path=latest_ckpt, resume_iter=start_iter)

        stats = state["stats"]
        seed = state.get("seed")

        return {
            "scheduler": scheduler,
            "archive": archive,
            "start_iter": start_iter,
            "stats": stats,
            "seed": seed
        }

    def setup_dask(self, batch_size=BATCH_SIZE, model = None, embedding_dim = None, model_path=None):
        """Create a Dask LocalCluster and scatter the evaluator to all workers.

        Takes either a pre-instantiated model and embedding_dim or a path to a pretrained model checkpoint.

        Returns ``(client, cluster, evaluator_future)``.
        """
        # dask.config.set({"distributed.worker.multiprocessing-method": "fork"})

        if model_path is not None:
            log.debug("Setting up Dask LocalCluster", n_workers=batch_size)
            cluster = LocalCluster(
                processes=True, n_workers=batch_size, threads_per_worker=1)
            client = Client(cluster)
            log.debug("Dask cluster ready", dashboard=client.dashboard_link)

            evaluator, model = EvaluatorMetrics.load_pretrained(model_path)
            evaluator_future = client.scatter(evaluator, broadcast=True)
            log.debug("Evaluator scattered to Dask workers", n_workers=batch_size)

            return client, cluster, evaluator_future, model

        elif model is not None:
            log.debug("Setting up Dask LocalCluster", n_workers=batch_size)
            cluster = LocalCluster(
                processes=True, n_workers=batch_size, threads_per_worker=1)
            client = Client(cluster)
            log.debug("Dask cluster ready", dashboard=client.dashboard_link)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            evaluator = EvaluatorMetrics(model, embedding_dim, device)

            evaluator_future = client.scatter(evaluator, broadcast=True)

            log.debug("Evaluator scattered to Dask workers", n_workers=batch_size)
            return client, cluster, evaluator_future, model

        else:
            raise ValueError("Either model or model_path must be provided for Dask setup.")

    @property
    def visualizer(self):
        """Access the ``ArchiveVisualizer`` for post-run plots and export."""
        return self._visualizer

    def run(self, total_iters, start_iter=None):
        """Execute the ask → evaluate → tell loop.

        Returns ``(global_best_score, global_best_id, stats)``.
        """
        if start_iter is None:
            start_iter = self.start_iter

        for i in range(start_iter, total_iters):
            
            do_recalc_threshold = i % RECALC_THRESHOLD_EVERY == 0
            population_iter = i < RANDOM_POPULATION_ITERS
            do_finetune = self.do_finetune and (i % RETRAIN_EVERY == 0)
            
            if not population_iter and do_recalc_threshold and i != start_iter:
                self._remap_archive(self.archive.data()["measures"])
                
            self.iteration = i  # for tracking in add() wrapper
            sols = self.scheduler.ask()
            sol_dicts = [array_to_solution(sol) for sol in sols]

            futs = [self.client.submit(_eval_on_worker, self.evaluator_future, sol)
                    for sol in sol_dicts]
            gathered = [f.result() for f in futs]

            score_list, clean_solutions = [], []
            # Measure is the embedding returned by the evaluator
            for (sol_id, ok, msg, objective_score, measure, phenotype_data), sol_dict in zip(gathered, sol_dicts):
                if not ok:
                    log.debug("Clamping bad score",
                                sol_id=sol_id, reason=msg)
                    objective_score = INVALID_SCORE
                else:
                    log.debug("Solution evaluated", sol_id=sol_id,
                             score=f"{objective_score:.2f}")
                    if objective_score > self.global_best_score:
                        self.global_best_score = objective_score
                        self.global_best_id = sol_id

                self._evaluation_buffer.record(
                    sol_id, sol_dict, measure, objective_score, ok, phenotype_data)
                clean_solutions.append((objective_score, measure))
                score_list.append(objective_score)

            score_batch, measures_batch = zip(*clean_solutions)

            # ── Tell with add-status tracking ──
            with self._track_add_status() as (new_insertions, substitutions):
                self.scheduler.tell(list(score_batch), list(measures_batch))

            new_insertions_dicts = [array_to_solution(s) for s in new_insertions]
            substitution_dicts   = [(array_to_solution(old), array_to_solution(new)) for old, new in substitutions]


            # Metrics and Visualizations:
            new_elites_count = len(new_insertions_dicts)   # was: counts["new"]
            sub_elites_count = len(substitution_dicts)
            batch_best = max(score_list) if score_list else INVALID_SCORE

            log.info(
                "Iteration complete",
                new_elites=new_elites_count,
                substituted=sub_elites_count,
                iteration=i,
                batch_best=f"{batch_best:.2f}",
                global_best=f"{self.global_best_score:.2f}",
                global_best_id=self.global_best_id,
            )

            data_archive = self.archive.data()
            arch_scores = data_archive["objective"]
            valid = arch_scores != INVALID_SCORE
            mean_val = np.mean(arch_scores[valid]) if np.any(valid) else 0.0
            best_val = np.max(arch_scores[valid]) if np.any(valid) else 0.0
            elites = self.archive.stats.num_elites
            log.info(
                "Archive stats",
                size=elites,
                mean=f"{mean_val:.2f}",
                best=f"{best_val:.2f}",
            )

            # ── Compute metrics ──
            measures = data_archive["measures"]

            qd_score = qd_stats.compute_qd_score(arch_scores)
            acceptance_rate = qd_stats.compute_acceptance_rate(new_elites_count, sub_elites_count, len(sol_dicts))
            mean_pairwise_dist = qd_stats.compute_mean_pairwise_dist(measures, seed=self.seed)
            high_quality_cov = qd_stats.compute_high_quality_coverage(arch_scores, threshold=10.0)
            mean_knn_novelty = qd_stats.compute_mean_knn_novelty(measures, k=NS_KNN)
            fitness_novelty_corr = qd_stats.compute_fitness_novelty_corr(measures, arch_scores, k=NS_KNN)


            self.stats.append({
                "iteration": i,
                "Archive size": elites,
                "iteration_best": batch_best,
                "global_best_score": self.global_best_score,
                "global_best_id": self.global_best_id,
                "new_elites": new_elites_count,
                "substituted_elites": sub_elites_count,
                "qd_score": qd_score,
                "acceptance_rate": acceptance_rate,
                "mean_pairwise_dist": mean_pairwise_dist,
                "high_quality_coverage": high_quality_cov,
                "mean_knn_novelty": mean_knn_novelty,
                "fitness_novelty_corr": fitness_novelty_corr,
            })

            log.info("Stats updated", iteration=i, stats=self.stats[-1])

            self._visualizer.plot_heatmap(i)
            self._visualizer.plot_grid(i, substitution_dicts)

            # Save plot_grid in stats
            self.stats[-1]["grid_state"] = self._visualizer.grid_state.copy()

            # Checkpoint, always at the end of the loop
            if i % CHECKPOINT_EVERY == 0 and i != start_iter:
                ckpt_name =  os.path.join(self.checkpoint_dir, f"checkpoint_{i:04d}.pkl")
                with open(ckpt_name, "wb") as f:
                    pickle.dump({
                        "scheduler": self.scheduler,
                        "seed": self.seed,
                        "iteration": i,
                        "stats": self.stats,
                    }, f)
                log.info("Checkpoint saved", path=ckpt_name, iteration=i)

                self._evaluation_buffer.save()

            if not population_iter and do_finetune and (i != start_iter):
                log.info("Retraining evaluator on current buffer elites and recalculating measures for all archived elites")
                ckpt_name =  os.path.join(self.checkpoint_dir, f"checkpoint_{i:04d}.pkl")
                with open(ckpt_name, "wb") as f:
                    pickle.dump({
                        "scheduler": self.scheduler,
                        "seed": self.seed,
                        "iteration": i,
                        "stats": self.stats,
                    }, f)
                log.info("Checkpoint saved", path=ckpt_name, iteration=i)
                self._evaluation_buffer.save()

                self._start_retraining_routine()


        # Final save
        ckpt_name = os.path.join(self.checkpoint_dir, f"checkpoint_{i:04d}.pkl")
        with open(ckpt_name, "wb") as f:
            pickle.dump({
                "scheduler": self.scheduler,
                "iteration": i,
                "stats": self.stats,
                "seed": self.seed,
            }, f)
        log.info("Checkpoint saved", path=ckpt_name, iteration=i)

        self._evaluation_buffer.save()
        return self.global_best_score, self.global_best_id, self.stats
    
    def _recalculate_novelty_threshold(self) -> float:
        """multiplicative proportional controller formula from "Unsupervised Behaviour Discovery
        with Quality-Diversity Optimisation" by Luca Grillotti and Antoine Cully 2022"""
        current_size = self.archive.stats.num_elites
        current_threshold = self.archive.novelty_threshold
        
        new_threshold = current_threshold * (1 + K_CSC * (current_size - TARGET_ARCHIVE_SIZE))
        
        new_threshold = max(new_threshold, 1e-3) # to avoid collapsing to zero if archive is empty
        return type(current_threshold)(new_threshold) # cast to the same type as current_threshold to avoid any issues

    def _remap_archive(self, new_measures):
        new_threshold = self._recalculate_novelty_threshold()
        log.info("Recalculating novelty threshold and remapping archive", new_threshold=new_threshold)
        current_elites = self.archive.data()
        solutions  = current_elites["solution"]
        objectives = current_elites["objective"]
        sol_to_add, obj_to_add, measure_to_add = self._filter_elites_by_novelty(
            solutions, objectives, new_measures, new_threshold
        )
        
        self.archive.clear()
        self._visualizer.reset_grid_state()
        self.archive._novelty_threshold = new_threshold # a bit of a hack to avoid reinitializing the archive just to update the threshold, since add() is the only method that uses it and we call it right after
        
        with self._track_add_status() as (new_insertions, substitutions):
            self.archive.add(sol_to_add, obj_to_add, measure_to_add)

        if len(new_insertions) != len(sol_to_add):
            raise ValueError(
                f"Expected all {len(sol_to_add)} remapped elites to be re-added "
                f"but only {len(new_insertions)} were accepted. "
                "Check the novelty-filter and add_status tracking."
            )
    
    def _start_retraining_routine(self):
        """Orchestrate one retraining cycle:
        finetune → remap measures → filter → rebuild archive → reset Dask.
        """
        current_elites = self.archive.data()
        solutions  = current_elites["solution"]
        elite_ids  = [array_to_solution(sol)["id"] for sol in solutions]
        elite_phenotype = self._evaluation_buffer.get_phenotype_data(elite_ids)

        finetuned_model, embedding_dim = self._finetune_embedding_model(elite_phenotype)
        new_measures = self._recompute_measures(finetuned_model, elite_phenotype)

        self._remap_archive(new_measures)

        finetuned_model.cpu()
        self._embedding_model = finetuned_model
        self.client.close()
        self.cluster.close()
        self.client, self.cluster, self.evaluator_future, _ = self.setup_dask(
            model=finetuned_model, embedding_dim=embedding_dim
        )

    def _finetune_embedding_model(self, elite_phenotype_data: list) -> tuple:
        """Build a mixed replay dataset and finetune a copy of the embedding model.

        Dataset composition
        -------------------
        * All valid entries in the evaluation buffer   (broad replay)
        * Elite entries repeated 2×                    (quality bias)

        The first ``FINETUNING_CONFIG["n_frozen_encoder_blocks"]`` encoder
        ResBlocks are frozen so generic low-level patterns from pretraining are
        preserved.

        Returns
        -------
        finetuned_model : MetricsVAE   (eval mode, on the training device)
        embedding_dim   : int
        """
        preprocessor = MetricsPreprocessor()

        def _safe_preprocess(raw):
            """Preprocess raw 7-col phenotype data; return None on failure."""
            try:
                return preprocessor(np.array(raw, dtype=np.float32))
            except Exception:
                return None

        valid_elites = [
            p
            for raw in elite_phenotype_data
            if raw is not None
            for p in (_safe_preprocess(raw),)
            if p is not None
        ]
        finetune_data = valid_elites
        
        if not finetune_data:
            raise ValueError("No valid phenotype data available for fine-tuning.")

        batch_size = min(_DC["batch_size"], len(valid_elites))
        log.info("Fine-tuning dataset built", total_samples=len(finetune_data), batch_size=batch_size)
        
        dataset = MetricsDataset(finetune_data)
        n_val   = max(1, int(_DC["val_split"] * len(dataset)))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_copy = copy.deepcopy(self._embedding_model)

        ft_config = TrainingConfig(
            finetune=True,
            lr=_FT["lr"],
            epochs=_FT["epochs"],
            patience=_FT["patience"],
            n_frozen_encoder_blocks=_FT["n_frozen_encoder_blocks"],
            n_cycles=_FT["kld"]["n_cycles"],
            max_beta=_FT["kld"]["max_beta"],
            ratio=_FT["kld"]["ratio"],
        )

        trainer = VAETrainer(model_copy, ft_config, device)
        trainer.fit(train_loader, val_loader)

        finetuned_model = trainer.model
        finetuned_model.eval()
        return finetuned_model, finetuned_model.latent_dim

    def _recompute_measures(self, model, phenotype_data_list: list) -> np.ndarray:
        """Re-encode a list of phenotype sequences with ``model``.

        Sequences that are ``None`` (failed evaluation) receive a zero vector
        so that index alignment with the archive is preserved.

        Valid sequences are encoded in mini-batches (size ``DATA_CONFIG["batch_size"]``)
        using the same padding + mask convention as the training DataLoader,
        avoiding OOM for large archives.

        Returns
        -------
        measures : np.ndarray, shape (N, latent_dim)
        """
        model.eval()
        device        = next(model.parameters()).device
        preprocessor  = MetricsPreprocessor()
        embedding_dim = model.latent_dim

        # Separate valid entries, preserving original indices
        valid_indices = []
        valid_tensors = []
        for i, p in enumerate(phenotype_data_list):
            if p is not None:
                arr = np.array(p, dtype=np.float32)
                arr = preprocessor(arr)
                valid_tensors.append(torch.tensor(arr, dtype=torch.float32))
                valid_indices.append(i)

        measures  = np.zeros((len(phenotype_data_list), embedding_dim), dtype=np.float32)

        batch_size = min(len(valid_tensors), _DC["batch_size"])
        with torch.no_grad():
            for start in range(0, len(valid_tensors), batch_size):
                batch_tensors = valid_tensors[start : start + batch_size]
                batch_indices = valid_indices[start : start + batch_size]

                # collate_fn pads to the longest sequence in this batch
                padded, mask = collate_fn(batch_tensors)
                padded = padded.to(device)
                mask   = mask.to(device)

                mu, _ = model.encode(padded, mask)
                mu_np = mu.cpu().numpy()

                for out_row, src_idx in enumerate(batch_indices):
                    measures[src_idx] = mu_np[out_row]

        return measures

    @staticmethod
    def _filter_elites_by_novelty(
        solutions: np.ndarray,
        objectives: np.ndarray,
        measures: np.ndarray,
        novelty_threshold: float = DEFAULT_ARCHIVE_THRESHOLD,
        k: int = NS_KNN,
    ) -> tuple[list, list, list]:
        """Filter archive elites after a measure-space change using KNN novelty.

        After the VAE is retrained, every elite's measure is recomputed in the
        new latent space.  This function decides which of those remapped elites
        are worth re-inserting:

        - **Novel** solutions (mean k-NN distance > *novelty_threshold*) are
          always admitted — they occupy a genuinely unique region.
        - **Non-novel** solutions compete locally: each one's nearest neighbour
          acts as an incumbent; within each group that shares the same
          nearest-neighbour, only the highest-fitness challenger is kept, and
          only if it beats that neighbour's own fitness.

        Parameters
        ----------
        solutions : np.ndarray, shape (N, D)
            Elite solution vectors (raw arrays, not dicts).
        objectives : np.ndarray, shape (N,)
            Fitness values corresponding to each elite.
        measures : np.ndarray, shape (N, M)
            Newly recomputed behaviour descriptors in the updated latent space.
        novelty_threshold : float
            Mean k-NN distance below which a solution is considered non-novel.
            Defaults to ARCHIVE_THRESHOLD; tune to match the scale of the new latent space.
        k : int
            Number of nearest neighbours used for novelty scoring.

        Returns
        -------
        sol_to_add : list of np.ndarray
            Filtered solution arrays ready to pass to ``archive.add``.
        obj_to_add : list of float
            Corresponding fitness values.
        measure_to_add : list of np.ndarray
            Corresponding measures in the new latent space.
        """
        
        if len(solutions) <= k:
            return list(solutions), list(objectives), list(measures)

        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(measures)
        distances, indices = nn.kneighbors(measures)

        # exclude self (column 0)
        novelty_scores       = distances[:, 1:].mean(axis=1)
        nearest_neighbor_idx = indices[:, 1]  # closest *other* solution

        novel_mask     = novelty_scores > novelty_threshold
        non_novel_mask = ~novel_mask

        sol_to_add: list     = []
        obj_to_add: list     = []
        measure_to_add: list = []

        # --- Pass 1: novel solutions, always admitted ---
        for i in np.where(novel_mask)[0]:
            sol_to_add.append(solutions[i])
            obj_to_add.append(objectives[i])
            measure_to_add.append(measures[i])

        # --- Pass 2: non-novel solutions — local winner-takes-all competition ---
        # Group solutions by the index of their nearest neighbour (the "cell" they
        # would collide in) and keep only the best challenger per group, provided
        # it outperforms that neighbour's own fitness.
        collision_groups: dict = {}   # nn_idx -> [(objective, solution_idx), ...]
        for i in np.where(non_novel_mask)[0]:
            nn_idx = nearest_neighbor_idx[i]
            collision_groups.setdefault(nn_idx, []).append((objectives[i], i))

        n_collision_winners = 0
        for nn_idx, competitors in collision_groups.items():
            best_obj, best_i = max(competitors, key=lambda x: x[0])
            if best_obj > objectives[nn_idx]:
                sol_to_add.append(solutions[best_i])
                obj_to_add.append(objectives[best_i])
                measure_to_add.append(measures[best_i])
                n_collision_winners += 1

        log.info(
            "Elite novelty filter complete",
            novel_admitted=int(novel_mask.sum()),
            collision_winners=n_collision_winners,
            total_to_add=len(sol_to_add),
        )
        return sol_to_add, obj_to_add, measure_to_add

    # -- archive add() tracking -----------------------------------------------

    @contextmanager
    def _track_add_status(self):
        substitutions = []   # list of (old_solution, new_solution)
        new_insertions = []  # list of new_solution
        original_store_add = self.archive._store.add

        def tracked_store_add(indices, data):
            indices_to_add = indices
            data_to_add = data

            occupied, old_data = self.archive._store.retrieve(indices)
            is_replacing = occupied
            is_new       = ~occupied

            # recheck to see if indices need to be replaced
            replacing_indices = indices[is_replacing]
            new_indices = indices[is_new]

            replacing_data_objectives = data["objective"][is_replacing]
            old_objectives = old_data["objective"][is_replacing]

            if np.any(replacing_data_objectives <= old_objectives):
                bad_mask = replacing_data_objectives <= old_objectives
                log.warning("Unexpectedly lower objective in add() call — skipping bad replacements",
                            indices=replacing_indices[bad_mask],
                            new_objectives=replacing_data_objectives[bad_mask],
                            old_objectives=old_objectives[bad_mask])

                # Drop only the bad replacements; keep valid ones and all new insertions
                repl_positions = np.where(is_replacing)[0]
                is_replacing[repl_positions[bad_mask]] = False
                valid_mask = is_replacing | is_new
                indices_to_add = indices[valid_mask]
                data_to_add = {k: v[valid_mask] for k, v in data.items()}

            if np.any(is_replacing):
                old_solutions = old_data["solution"][is_replacing].copy()
                new_solutions = data["solution"][is_replacing]
                substitutions.extend(zip(old_solutions, new_solutions))

            if np.any(is_new):
                new_insertions.extend(data["solution"][is_new])

            # log new insertions_ids and substitution_ids for this add() call
            # log.trace("New insertions: ", new= [array_to_solution(s)["id"] for s in new_insertions])
            # log.trace("Substitutions: ", sub= [(array_to_solution(old)["id"], array_to_solution(new)["id"]) for old, new in substitutions])
            if len(indices_to_add) > 0:
                original_store_add(indices_to_add, data_to_add)

        self.archive._store.add = tracked_store_add
        try:
            yield new_insertions, substitutions
        finally:
            if "add" in self.archive._store.__dict__:
                del self.archive._store.__dict__["add"]
