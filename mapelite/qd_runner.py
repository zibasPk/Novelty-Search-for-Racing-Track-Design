# qd_runner.py
# Shared infrastructure for Quality-Diversity search loops.
# Both novelty_search.ipynb and CVT_mapelite.ipynb delegate to these classes.

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
    RunMode
)
from mapelite.evaluator import EvaluatorMetrics
from mapelite.archive_visualizer import ArchiveVisualizer
from dask.distributed import Client, LocalCluster
import numpy as np
from numpy_groupies import aggregate_nb as aggregate
import os
import glob
import pickle
import datetime
import json
import joblib
from mapelite.logging_config import get_logger
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
        client,
        evaluator_future,
        checkpoint_dir,
        heatmap_dir,
        gridplot_dir,
        start_iter=DEFAULT_START_ITER,
        buffer_path=None,
        seed=None,
        centroids=None,
        initial_WSS=None,
        stats=None,
        grid_state=None
    ):
        self.scheduler = scheduler
        self.archive = archive
        self.client = client
        self.evaluator_future = evaluator_future
        self.stats: list[dict] = stats if stats is not None else []

        self.start_iter = start_iter

        self.checkpoint_dir = checkpoint_dir
        self.heatmap_dir = heatmap_dir
        self.gridplot_dir = gridplot_dir
        self.buffer_path = buffer_path or os.path.join(
            BUFFER_DIR, "buffer.json")

        self.seed = seed
        # Fixed centroids for WSS (CVT case). None → use archive measures (NS case).
        self.centroids = np.asarray(
            centroids) if centroids is not None else None
        self.initial_WSS = initial_WSS
        # Mutable run state
        self.global_best_score = stats[0].get("global_best_score", INVALID_SCORE) if stats else INVALID_SCORE
        self.global_best_id = stats[0].get("global_best_id") if stats else None

        # Evaluation buffer & visualizer
        self._evaluation_buffer = EvaluationBuffer(self.buffer_path)
        self._visualizer = ArchiveVisualizer(
            archive, self.stats, heatmap_dir, gridplot_dir, seed=seed, grid_state=grid_state)

        self._run_mode = RunMode.CVT if centroids is not None else RunMode.NS
        self.iteration = 0

    @classmethod
    def load_state(cls, state, client, evaluator_future, checkpoint_dir, heatmap_dir, gridplot_dir, buffer_path, seed):
        instance = cls(
            scheduler=state["scheduler"],
            archive=state["archive"],
            start_iter=state["start_iter"],
            stats=state["stats"],
            evaluator_future=evaluator_future,
            client=client,
            checkpoint_dir=checkpoint_dir,
            heatmap_dir=heatmap_dir,
            gridplot_dir=gridplot_dir,
            buffer_path=buffer_path,
            seed=seed,
            centroids=getattr(state["archive"], "centroids", None),
            initial_WSS=state["stats"][0].get("initial_WSS", None),
            grid_state=state["stats"][-1].get("grid_state", None),
        )
        return instance

    @staticmethod
    def setup_dask(batch_size=BATCH_SIZE, model = None, embedding_dim = None, model_path=None):
        """Create a Dask LocalCluster and scatter the evaluator to all workers.
        
        Takes either a pre-instantiated model and embedding_dim or a path to a pretrained model checkpoint.

        Returns ``(client, cluster, evaluator_future)``.
        """
        if model_path is not None:
            log.debug("Setting up Dask LocalCluster", n_workers=batch_size)
            cluster = LocalCluster(
                processes=True, n_workers=batch_size, threads_per_worker=1)
            client = Client(cluster)
            log.debug("Dask cluster ready", dashboard=client.dashboard_link)

            evaluator = EvaluatorMetrics.load_pretrained(model_path)
            evaluator_future = client.scatter(evaluator, broadcast=True)
            log.debug("Evaluator scattered to Dask workers", n_workers=batch_size)

            return client, cluster, evaluator_future
        
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
            return client, cluster, evaluator_future
        
        else:
            raise ValueError("Either model or model_path must be provided for Dask setup.")

    @staticmethod
    def get_state_from_checkpoint(checkpoint_dir):
        """Check for existing checkpoints and return the latest state if found."""
        
        checkpoints = sorted(glob.glob(f"{checkpoint_dir}checkpoint_*.pkl"))

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
        
    def _start_retraining_routine(self):
        # get current elites from archive
        # finetune the VAE on these elites
        # recalculate measures for all elites in archive with the updated VAE
        # update the archive with the new measures (without changing fitness or solutions)
        # update the evaluator_future with the new evaluator (with updated VAE)
        
        current_elites = self.archive.data()
        elite_ids = [array_to_solution(sol)["id"] for sol in current_elites["solution"]]
        phenotype_data = self._evaluation_buffer.get_phenotype_data(elite_ids)
      
  
        
    def run(self, total_iters, start_iter=None):
        """Execute the ask → evaluate → tell loop.

        Returns ``(global_best_score, global_best_id, stats)``.
        """
        if start_iter is None:
            start_iter = self.start_iter

        for i in range(start_iter, total_iters):
            
            if (i % RETRAIN_EVERY == 0) and (i != start_iter):
                log.info("Retraining evaluator on current buffer elites and recalculating measures for all archived elites")
                self._start_retraining_routine()
                
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
                    log.info("Clamping bad score",
                                sol_id=sol_id, reason=msg)
                    objective_score = INVALID_SCORE
                else:
                    log.info("Solution evaluated", sol_id=sol_id,
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
            
            qd_score = self._compute_qd_score(arch_scores)
            acceptance_rate = self._compute_acceptance_rate(new_elites_count, sub_elites_count, len(sol_dicts))
            mean_pairwise_dist = self._compute_mean_pairwise_dist(measures)
            high_quality_cov = self._compute_high_quality_coverage(arch_scores, threshold=10.0)
            
            mean_knn_novelty = None
            fitness_novelty_corr = None
            wss = None
            
            match self._run_mode:
                case RunMode.CVT:
                    wss = self._compute_wss(self.centroids) 
                case RunMode.NS:
                    mean_knn_novelty = self._compute_mean_knn_novelty(measures, k = NS_KNN)
                    fitness_novelty_corr = self._compute_fitness_novelty_corr(measures, arch_scores, k = NS_KNN)
                case _:
                    log.warning("Unknown run mode: ", mode=self._run_mode)
            
            
            self.stats.append({
                "iteration": i,
                "initial_WSS": self.initial_WSS,
                "Archive size": elites,
                "iteration_best": batch_best,
                "global_best_score": self.global_best_score,
                "global_best_id": self.global_best_id,
                "new_elites": new_elites_count,
                "substituted_elites": sub_elites_count,
                "wss": wss,
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
                ckpt_name = f"{self.checkpoint_dir}checkpoint_{i:04d}.pkl"
                with open(ckpt_name, "wb") as f:
                    pickle.dump({
                        "scheduler": self.scheduler,
                        "seed": self.seed,
                        "iteration": i,
                        "stats": self.stats,
                    }, f)
                log.info("Checkpoint saved", path=ckpt_name, iteration=i)
                
                self._evaluation_buffer.save()


        # Final save
        ckpt_name = f"{self.checkpoint_dir}checkpoint_{i:04d}.pkl"
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

    @property
    def visualizer(self):
        """Access the ``ArchiveVisualizer`` for post-run plots and export."""
        return self._visualizer
    
    
    
    # -- metrics helpers -----------------------------------------------------------

    def _compute_wss(self, centroids: np.ndarray) -> float:
        """Mean per-point Within-Cluster Sum of Squares for all buffer embeddings.

        For each embedding in the buffer the squared distance to the nearest
        centroid is computed; returns the *mean* of these minimum squared
        distances (total WSS / N) so the value is comparable across iterations
        regardless of how many tracks have been evaluated.

        Parameters
        ----------
        centroids : np.ndarray, shape (K, D)
            Cluster centres to use.
        """
        if len(self._evaluation_buffer.entries) == 0 or len(centroids) == 0:
            return float("nan")

        embeddings = np.array(
            [e["measure"] for e in self._evaluation_buffer.entries],
            dtype=float,
        )  # (N, D)
        centroids = np.asarray(centroids, dtype=float)  # (K, D)

        # Pairwise squared distances via broadcasting: (N, K)
        sq_dists = np.sum(
            (embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        return float(np.mean(np.min(sq_dists, axis=1)))

    def _compute_qd_score(self, objective_scores: np.ndarray) -> float:
        """Sum of all valid elite fitnesses (QD-Score)."""
        valid = objective_scores[(objective_scores != INVALID_SCORE) & np.isfinite(objective_scores)]
        return float(np.sum(valid)) if len(valid) > 0 else 0.0

    def _compute_acceptance_rate(self, new_count: int, sub_count: int, batch_size: int) -> float:
        """Fraction of evaluated candidates accepted (new or improved) into the archive."""
        return (new_count + sub_count) / batch_size if batch_size > 0 else 0.0

    def _compute_mean_pairwise_dist(self, measures: np.ndarray) -> float:
        """Mean pairwise Euclidean distance among archive members (sampled for speed)."""
        n = len(measures)
        if n < 2:
            return float("nan")
        max_sample = 500
        if n > max_sample:
            rng = np.random.default_rng(self.seed)
            sample = measures[rng.choice(n, max_sample, replace=False)]
        else:
            sample = measures
        # Use |a-b|^2 = |a|^2 + |b|^2 - 2*a·b for memory efficiency
        sq_norms = np.sum(sample ** 2, axis=1)
        sq_dists = np.clip(
            sq_norms[:, None] + sq_norms[None, :] - 2 * (sample @ sample.T),
            0, None,
        )
        i_upper = np.triu_indices(len(sample), k=1)
        return float(np.mean(np.sqrt(sq_dists[i_upper])))

    def _compute_high_quality_coverage(self, objective_scores: np.ndarray,
                                       threshold: float = 20.0) -> int:
        """Count of archive elites with fitness >= *threshold*."""
        valid = objective_scores[(objective_scores != INVALID_SCORE) & np.isfinite(objective_scores)]
        return int(np.sum(valid >= threshold))

    def _compute_mean_knn_novelty(self, measures: np.ndarray, k: int = 5) -> float:
        """Mean k-NN distance among archive members — the NS novelty proxy."""
        if len(measures) < k + 1:
            return float("nan")
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(measures)
        dists, _ = nbrs.kneighbors(measures)
        return float(np.mean(dists[:, 1:]))  # exclude self (column 0)

    def _compute_fitness_novelty_corr(self, measures: np.ndarray,
                                      objective_scores: np.ndarray, k: int = 5) -> float:
        """Pearson correlation between per-elite k-NN novelty score and fitness."""
        valid_mask = (objective_scores != INVALID_SCORE) & np.isfinite(objective_scores)
        m = measures[valid_mask]
        f = objective_scores[valid_mask]
        if len(m) < k + 2:
            return float("nan")
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(m)
        dists, _ = nbrs.kneighbors(m)
        novelty = np.mean(dists[:, 1:], axis=1)
        if np.std(novelty) < 1e-10 or np.std(f) < 1e-10:
            return float("nan")
        return float(np.corrcoef(novelty, f)[0, 1])

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
                log.warning("Unexpectedly lower objective in add() call",
                            indices=replacing_indices[replacing_data_objectives <= old_objectives],
                            new_objectives=replacing_data_objectives[replacing_data_objectives <= old_objectives],
                            old_objectives=old_objectives[replacing_data_objectives <= old_objectives])
                
                is_replacing = False
                indices_to_add = new_indices  # only add the new indices, skip the replacements
                data_to_add = {k: v[is_new] for k, v in data.items()}
                
            if np.any(is_replacing):
                old_solutions = old_data["solution"][is_replacing].copy()
                new_solutions = data["solution"][is_replacing]
                substitutions.extend(zip(old_solutions, new_solutions))

            if np.any(is_new):
                new_insertions.extend(data["solution"][is_new])

            # log new insertions_ids and substitution_ids for this add() call
            log.debug("New insertions: ", new= [array_to_solution(s)["id"] for s in new_insertions])
            log.debug("Substitutions: ", sub= [(array_to_solution(old)["id"], array_to_solution(new)["id"]) for old, new in substitutions])
            if len(indices_to_add) > 0:
                original_store_add(indices_to_add, data_to_add)
        
        self.archive._store.add = tracked_store_add
        try:
            yield new_insertions, substitutions
        finally:
            if "add" in self.archive._store.__dict__:
                del self.archive._store.__dict__["add"]


