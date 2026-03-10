# qd_runner.py
# Shared infrastructure for Quality-Diversity search loops.
# Both novelty_search.ipynb and CVT_mapelite.ipynb delegate to these classes.

import os
import glob
import pickle
import datetime
import json
import joblib
import umap
import requests
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from ribs.archives import AddStatus
from dask.distributed import Client, LocalCluster

from mapelite.evaluator import EvaluatorMetrics
from mapelite.config import (
    BASE_URL,
    BATCH_SIZE,
    BUFFER_DIR,
    CHECKPOINT_EVERY,
    INVALID_SCORE,
    HEATMAP_DIR
)
from utils import array_to_solution


# ── Evaluation Buffer ───────────────────────────────────────────────────────

class EvaluationBuffer:
    """Accumulates every evaluated track (id, spline data, embedding) and
    persists them to a JSON file in ``data/buffers/``.

    The buffer is append-only: on resume it loads existing entries and
    continues appending.  It is saved at every checkpoint interval and
    once more when the loop finishes.
    """

    def __init__(self, buffer_path: str):
        self.buffer_path = buffer_path
        os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
        self.entries: list[dict] = []
        self._load()

    # -- persistence ----------------------------------------------------------

    def _load(self):
        """Resume from an existing buffer file, if present."""
        if os.path.exists(self.buffer_path):
            with open(self.buffer_path, "r") as f:
                data = json.load(f)
            self.entries = data.get("tracks", [])
            print(
                f"[Buffer] Resumed {len(self.entries)} entries from {self.buffer_path}")
        else:
            print(f"[Buffer] No existing buffer found — starting empty.")

    def save(self):
        """Write the full buffer to disk."""
        payload = {
            "total": len(self.entries),
            "timestamp": datetime.datetime.now().isoformat(),
            "tracks": self.entries,
        }
        with open(self.buffer_path, "w") as f:
            json.dump(payload, f)
        print(
            f"[Buffer] Saved {len(self.entries)} entries to {self.buffer_path}")

    # -- recording ------------------------------------------------------------

    def record(self, sol_id, sol_dict: dict, embedding, score: float, ok: bool):
        """Record a single evaluated track.

        Parameters
        ----------
        sol_id : float | int
            Unique solution identifier.
        sol_dict : dict
            Solution dictionary produced by ``array_to_solution`` containing
            ``dataSet`` and ``selectedCells`` (the data needed to reconstruct
            the spline).
        embedding : array-like
            Behavioural descriptor / embedding vector.
        score : float
            Fitness score returned by the evaluator.
        ok : bool
            Whether the evaluation succeeded.
        """
        entry = {
            "id": sol_id,
            "dataSet": sol_dict.get("dataSet", []),
            "selectedCells": sol_dict.get("selectedCells", []),
            "mode": sol_dict.get("mode", ""),
            "rngMode": sol_dict.get("rngMode", "uniform"),
            "embedding": np.asarray(embedding).tolist(),
            "score": float(score),
            "valid": ok,
        }
        self.entries.append(entry)

    def __len__(self):
        return len(self.entries)


# ── Archive Visualizer ──────────────────────────────────────────────────────

class ArchiveVisualizer:
    """Groups all archive visualization and export methods.

    Parameters
    ----------
    archive : pyribs archive
        The QD archive to visualize.
    stats : list[dict]
        Stats list produced by ``QDRunner.run``.
    heatmap_dir : str
        Directory to save UMAP heatmap PNGs.
    gridplot_dir : str
        Directory to save grid plot PNGs.
    seed : int, optional
        Random seed for UMAP reproducibility.
    """

    def __init__(self, archive, stats, heatmap_dir, gridplot_dir, seed=None):
        self.archive = archive
        self.stats = stats
        self.heatmap_dir = heatmap_dir
        self.gridplot_dir = gridplot_dir
        self.seed = seed
        self._track_cache: dict = {}
        self._umap_model = joblib.load("mapelite\embeddings\models\model_metrics_VAE\model_metrics_VAE_latent32_umap.joblib")

    # -- track outline helper -------------------------------------------------

    def _get_track_outline(self, sol_dict):
        """Call /reconstruct for *sol_dict* and return ``(xs, ys)`` float arrays.

        Results are cached by solution ID so the same solution is never
        fetched more than once across animation frames.
        Returns ``None`` on any error or if the server is unavailable.
        """
        sol_id = sol_dict.get("id")
        if sol_id in self._track_cache:
            return self._track_cache[sol_id]

        result = None
        try:
            resp = requests.post(
                f"{BASE_URL}/reconstruct",
                json={
                    "mode": sol_dict["mode"],
                    "dataSet": sol_dict["dataSet"],
                    "selectedCells": sol_dict.get("selectedCells", []),
                },
                timeout=5,
            )
            resp.raise_for_status()
            track = resp.json().get("track", [])
            if track:
                xs = np.array([p["x"] for p in track], dtype=float)
                ys = np.array([p["y"] for p in track], dtype=float)
                result = (xs, ys)
        except Exception:
            pass

        self._track_cache[sol_id] = result
        return result

    # -- grid plot ------------------------------------------------------------

    def plot_grid(self, iteration_idx=None, max_cols=15, max_sub_color=20,
                  max_rows=None, save_dir=None):
        """Render archive buckets as a 2D grid colored by substitution count.

        Parameters
        ----------
        iteration_idx : int, optional
            Index into *stats* to visualize.  ``None`` → last iteration.
        max_cols : int
            Fixed number of columns in the grid.
        max_sub_color : int
            Fixed upper bound for the colorbar (substitution count).
        max_rows : int, optional
            Fixed number of rows.  ``None`` → derived from the total number
            of unique buckets across *all* stats entries.
        save_dir : str, optional
            Directory where the figure should be saved.  Falls back to
            ``self.gridplot_dir`` if *None*.
        """
        if save_dir is None:
            save_dir = self.gridplot_dir

        stats = self.stats
        archive = self.archive

        if iteration_idx is None:
            iteration_idx = len(stats) - 1

        # ── Fixed grid dimensions derived from the full stats list ──
        cols = max_cols
        if max_rows is None:
            seen = set()
            total_unique = 0
            for s in stats:
                for idx in s.get("new_bucket_indices", []):
                    if idx not in seen:
                        seen.add(idx)
                        total_unique += 1
            max_rows = max(1, int(np.ceil(total_unique / cols)))

        # Replay stats up to iteration_idx
        bucket_order = []
        sub_counts = {}
        for s in stats[:iteration_idx + 1]:
            for idx in s.get("new_bucket_indices", []):
                bucket_order.append(idx)
            for idx in s.get("substituted_bucket_indices", []):
                sub_counts[idx] = sub_counts.get(idx, 0) + 1

        if len(bucket_order) == 0:
            return

        # Build grid — always (max_rows × cols), NaN = not yet filled
        grid = np.full((max_rows, cols), np.nan)
        for pos, idx in enumerate(bucket_order):
            r, c = divmod(pos, cols)
            grid[r, c] = sub_counts.get(idx, 0)

        cmap = LinearSegmentedColormap.from_list(
            "sub", ["white", "#ffff00", "red"])
        cmap.set_bad("lightgray")

        fig_w = max(6, cols * 0.6)
        fig_h = max(4, max_rows * 0.6)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=max_sub_color,
                       aspect="equal", interpolation="nearest",
                       origin="upper",
                       extent=(-0.5, cols - 0.5, max_rows - 0.5, -0.5))

        plt.colorbar(im, ax=ax, label="Substitution Count",
                     fraction=0.046, pad=0.04)

        # ── Build bucket → solution map for track drawing ──
        bucket_to_sol = {}
        if archive is not None:
            arch_data = archive.data()
            for _idx, _sol in zip(arch_data["index"], arch_data["solution"]):
                bucket_to_sol[int(_idx)] = array_to_solution(_sol)

        # Buckets newly inserted or substituted at this specific iteration
        current_changed = set(stats[iteration_idx].get("new_bucket_indices", [])) | \
                          set(stats[iteration_idx].get("substituted_bucket_indices", []))

        for pos, idx in enumerate(bucket_order):
            r, c = divmod(pos, cols)
            count = sub_counts.get(idx, 0)

            if idx in bucket_to_sol:
                outline = self._get_track_outline(bucket_to_sol[idx])
                if outline is not None:
                    xs, ys = outline
                    xspan = xs.max() - xs.min()
                    yspan = ys.max() - ys.min()
                    if xspan > 1e-6 and yspan > 1e-6:
                        pad = 0.1
                        xs_n = (xs - xs.min()) / xspan
                        ys_n = (ys - ys.min()) / yspan
                        cell_xs = c - (0.5 - pad) + xs_n * (1.0 - 2 * pad)
                        cell_ys = r - (0.5 - pad) + ys_n * (1.0 - 2 * pad)
                        track_color = "red" if idx in current_changed else "black"
                        track_lw = 0.9 if idx in current_changed else 0.4
                        ax.plot(cell_xs, cell_ys, color=track_color,
                                linewidth=track_lw, alpha=0.55, zorder=2)

            if count > 0:
                ax.text(c - 0.42, r - 0.40, str(count),
                        ha="left", va="top", fontsize=5, zorder=3,
                        color="black" if count < max_sub_color * 0.6 else "white")

        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, max_rows, 1), minor=True)
        ax.grid(which="minor", color="gray", linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(max_rows - 0.5, -0.5)

        it_label = stats[iteration_idx]["iteration"]
        ax.set_title(f"Archive Grid — Iteration {it_label}  "
                     f"({len(bucket_order)} / {max_rows * cols} buckets filled)")

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"archive_grid_iter_{iteration_idx:04d}.png"),
                        dpi=100, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

    # -- heatmap --------------------------------------------------------------

    def plot_heatmap(self, iteration, save_dir=None, starting_size = ((-1, 8.5), (7.5, 11))):
        """2D UMAP compression of the archive's behavioural space, colored by fitness."""
        # Min/max fitness for consistent coloring across iterations.  Adjust as needed.
        MAX_FIT = 60
        MIN_FIT = 0
        
        if save_dir is None:
            save_dir = self.heatmap_dir

        archive_data = self.archive.data()
        embeddings = archive_data["measures"]
        fitnesses = archive_data["objective"]

        min_samples = 16
        if len(embeddings) < min_samples:
            return

        solutions = archive_data["solution"]
        solution_dicts = [array_to_solution(sol) for sol in solutions]

        umap_compression = self._umap_model.transform(embeddings)
        fitnesses = np.asarray(fitnesses)


        cmap = plt.get_cmap("viridis").copy()
        # Set out of range values to distinct colors: under=below MIN_FIT, over=above MAX_FIT
        cmap.set_under("deeppink")
        cmap.set_over("red")

        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))

        sc = ax.scatter(umap_compression[:, 0], umap_compression[:, 1],
                        c=fitnesses, cmap=cmap, s=10, vmin=MIN_FIT, vmax=MAX_FIT)
        plt.colorbar(sc, ax=ax, label="Fitness", extend="both")

        x_min = min(umap_compression[:, 0].min(), starting_size[0][0])
        x_max = max(umap_compression[:, 0].max(), starting_size[0][1])
        y_min = min(umap_compression[:, 1].min(), starting_size[1][0])
        y_max = max(umap_compression[:, 1].max(), starting_size[1][1])

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"archive_heatmap_iter_{iteration}.png"))
        plt.close(fig)

        np.savez_compressed(os.path.join(save_dir, f"archive_data_iter_{iteration}.npz"),
                            umap=umap_compression, fitness=fitnesses, solutions=solution_dicts)

    # -- stats plot -----------------------------------------------------------

    def plot_stats(self, title="QD Run Statistics"):
        """6-row run-statistics plot."""
        stats = self.stats

        iterations = [s["iteration"] for s in stats]
        archive_sizes = [s["Archive size"] for s in stats]
        iter_best = [s["iteration_best"] for s in stats]
        global_best = [s["global_best_score"] for s in stats]
        new_elites = [s["new_elites"] for s in stats]
        sub_elites = [s["substituted_elites"] for s in stats]
        wss_values = [s.get("wss", float("nan")) for s in stats]

        iter_best_clean = [v if v != INVALID_SCORE else np.nan for v in iter_best]

        fig, axes = plt.subplots(6, 1, figsize=(14, 19), sharex=True)
        fig.suptitle(f"{title} — Run Statistics", fontsize=16, fontweight="bold")

        ax = axes[0]
        ax.plot(iterations, archive_sizes, color="tab:blue", linewidth=1.5)
        ax.set_ylabel("Archive Size")
        ax.set_title("Archive Growth")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(iterations, iter_best_clean, label="Iteration Best",
                color="tab:orange", alpha=0.6, linewidth=1)
        ax.plot(iterations, global_best, label="Global Best",
                color="tab:red", linewidth=2)
        ax.set_ylabel("Fitness Score")
        ax.set_title("Fitness Progress")
        ax.legend()
        ax.grid(True, alpha=0.3)

        bar_width = max(1, len(iterations) // 200)

        ax = axes[2]
        ax.bar(iterations, new_elites, width=bar_width,
               color="tab:red", alpha=0.8)
        ax.set_ylabel("Count")
        ax.set_title("New Elites per Iteration")
        ax.grid(True, alpha=0.3)

        ax = axes[3]
        ax.bar(iterations, sub_elites, width=bar_width,
               color="tab:blue", alpha=0.8)
        ax.set_ylabel("Count")
        ax.set_title("Substituted Elites per Iteration")
        ax.grid(True, alpha=0.3)

        ax = axes[4]
        cum_new = np.cumsum(new_elites)
        cum_sub = np.cumsum(sub_elites)
        ax.plot(iterations, cum_new, label="Cumulative New",
                color="tab:green", linewidth=1.5)
        ax.plot(iterations, cum_sub, label="Cumulative Substituted",
                color="tab:purple", linewidth=1.5)
        ax.plot(iterations, cum_new + cum_sub, label="Cumulative Total",
                color="tab:blue", linewidth=2, linestyle="--")
        ax.set_ylabel("Cumulative Count")
        ax.set_title("Cumulative Elite Insertions")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[5]
        ax.plot(iterations, wss_values, color="tab:brown", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("WSS")
        ax.set_title("Within-Cluster Sum of Squares (WSS)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

        print(f"\n{'='*50}")
        print(f"  {title} Summary")
        print(f"{'='*50}")
        print(f"  Total iterations:        {len(stats)}")
        print(f"  Final archive size:      {archive_sizes[-1]}")
        print(f"  Global best fitness:     {global_best[-1]:.4f}")
        print(f"  Total new elites:        {sum(new_elites)}")
        print(f"  Total substituted:       {sum(sub_elites)}")
        print(f"  Avg new elites/iter:     {np.mean(new_elites):.2f}")
        print(f"  Avg substituted/iter:    {np.mean(sub_elites):.2f}")
        print(f"{'='*50}")

    # -- elite export ---------------------------------------------------------

    def export_elites(self, output_path, algorithm_label, seed,
                      global_best_score, global_best_id):
        """Save all valid elites to a JSON file for reconstruction & visualization."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        data_archive = self.archive.data()
        solutions = np.array(data_archive["solution"])
        objectives = np.array(data_archive["objective"])
        measures = np.array(data_archive["measures"])

        elites_list = []
        skipped_invalid = 0
        for idx in range(len(objectives)):
            fit = float(objectives[idx])
            if not np.isfinite(fit) or fit == INVALID_SCORE:
                skipped_invalid += 1
                continue

            sol_arr = solutions[idx]
            sol_dict = array_to_solution(sol_arr)

            elite = {
                "id":            sol_dict["id"],
                "mode":          sol_dict["mode"],
                "rngMode":       sol_dict.get("rngMode", "uniform"),
                "dataSet":       sol_dict["dataSet"],
                "selectedCells": sol_dict["selectedCells"],
                "trackSize":     len(sol_dict["selectedCells"]),
                "fitness":       fit,
                "embedding":     measures[idx].tolist(),
                "archiveIndex":  int(idx),
            }
            elites_list.append(elite)

        if skipped_invalid:
            print(
                f"Skipped {skipped_invalid} elites with invalid fitness (INVALID_SCORE or NaN)")

        elites_list.sort(key=lambda e: e["fitness"], reverse=True)

        output = {
            "metadata": {
                "algorithm":    algorithm_label,
                "totalElites":  len(elites_list),
                "embeddingDim": int(measures.shape[1]),
                "solutionDim":  int(solutions.shape[1]),
                "seed":         seed,
                "iterations":   len(self.stats),
                "globalBest":   float(global_best_score),
                "globalBestId": global_best_id,
                "timestamp":    datetime.datetime.now().isoformat(),
            },
            "elites": elites_list,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Saved {len(elites_list)} elites to {output_path}")
        print(
            f"  Best fitness:  {elites_list[0]['fitness']:.4f} (ID={elites_list[0]['id']})")
        print(f"  Worst fitness: {elites_list[-1]['fitness']:.4f}")
        print(f"  File size:     {os.path.getsize(output_path) / 1024:.1f} KB")


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
        stats_path,
        buffer_path=None,
        seed=None,
        centroids=None,
    ):
        self.scheduler = scheduler
        self.archive = archive
        self.client = client
        self.evaluator_future = evaluator_future

        self.checkpoint_dir = checkpoint_dir
        self.heatmap_dir = heatmap_dir
        self.gridplot_dir = gridplot_dir
        self.stats_path = stats_path
        self.buffer_path = buffer_path or os.path.join(BUFFER_DIR, "buffer.json")
        self.seed = seed
        # Fixed centroids for WSS (CVT case). None → use archive measures (NS case).
        self.centroids = np.asarray(centroids) if centroids is not None else None

        # Mutable run state
        self.global_best_score = INVALID_SCORE
        self.global_best_id = None
        self.stats: list[dict] = []

        # Per-bucket tracking (rebuilt on resume)
        self._bucket_order: list[int] = []
        self._sub_counts: dict[int, int] = {}

        # Evaluation buffer & track cache
        self._evaluation_buffer = EvaluationBuffer(self.buffer_path)
        self._visualizer = ArchiveVisualizer(
            archive, self.stats, heatmap_dir, gridplot_dir, seed=seed)

    # -- factory helpers ------------------------------------------------------

    @staticmethod
    def setup_dask(batch_size=BATCH_SIZE):
        """Create a Dask LocalCluster and scatter the evaluator to all workers.

        Returns ``(client, cluster, evaluator_future)``.
        """
        print("Setting up Dask LocalCluster...")
        cluster = LocalCluster(
            processes=True, n_workers=batch_size, threads_per_worker=1)
        client = Client(cluster)
        print(f"Dask Dashboard link: {client.dashboard_link}")

        evaluator = EvaluatorMetrics.load_pretrained(
            "mapelite/embeddings/models/model_metrics_VAE/model_metrics_VAE_latent32.pth")
        evaluator_future = client.scatter(evaluator, broadcast=True)
        print(f"Evaluator scattered to {batch_size} Dask workers")

        return client, cluster, evaluator_future

    @staticmethod
    def resume_from_checkpoint(checkpoint_dir, stats_path):
        """Try to restore scheduler & stats from the latest checkpoint.

        Returns a dict with keys: ``scheduler``, ``archive``, ``start_iter``,
        ``global_best_score``, ``global_best_id``, ``stats``.
        ``scheduler`` / ``archive`` are *None* when no checkpoint was found.
        """
        checkpoints = sorted(glob.glob(f"{checkpoint_dir}checkpoint_*.pkl"))
        start_iter = 1
        global_best_score = INVALID_SCORE
        global_best_id = None
        scheduler = None
        archive = None
        stats = []

        if checkpoints:
            latest_ckpt = checkpoints[-1]
            with open(latest_ckpt, "rb") as f:
                state = pickle.load(f)
            scheduler = state["scheduler"]
            archive = scheduler.archive
            start_iter = state["iteration"] + 1
            global_best_score = state["global_best_score"]
            global_best_id = state["global_best_id"]
            print(
                f"[Resume] Loaded {latest_ckpt}, resuming from iteration {start_iter}")
        else:
            print("[Resume] No checkpoint found — starting fresh.")

        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                stats = pickle.load(f)
        print(f"[Resume] Resumed stats with {len(stats)} entries")

        return {
            "scheduler": scheduler,
            "archive": archive,
            "start_iter": start_iter,
            "global_best_score": global_best_score,
            "global_best_id": global_best_id,
            "stats": stats,
        }

    def load_state(self, start_iter=1, global_best_score=None,
                   global_best_id=None, stats=None):
        """Restore mutable run state (typically from ``resume_from_checkpoint``).

        Returns *self* for chaining.
        """
        if global_best_score is not None:
            self.global_best_score = global_best_score
        if global_best_id is not None:
            self.global_best_id = global_best_id
        if stats is not None:
            self.stats = stats
            # Keep the visualizer's stats reference in sync
            self._visualizer.stats = self.stats

        # Rebuild per-bucket tracking from loaded stats
        self._bucket_order = []
        self._sub_counts = {}
        for s in self.stats:
            for idx in s.get("new_bucket_indices", []):
                self._bucket_order.append(idx)
            for idx in s.get("substituted_bucket_indices", []):
                self._sub_counts[idx] = self._sub_counts.get(idx, 0) + 1

        return self

    # -- WSS helper -----------------------------------------------------------

    def _compute_wss(self, centroids: np.ndarray) -> float:
        """Within-cluster Sum of Squares for all buffer embeddings.

        For each embedding in the buffer the squared distance to the nearest
        centroid is computed; WSS is the sum of these minimum squared distances.

        Parameters
        ----------
        centroids : np.ndarray, shape (K, D)
            Cluster centres to use.
        """
        if len(self._evaluation_buffer.entries) == 0 or len(centroids) == 0:
            return float("nan")

        embeddings = np.array(
            [e["embedding"] for e in self._evaluation_buffer.entries],
            dtype=float,
        )  # (N, D)
        centroids = np.asarray(centroids, dtype=float)  # (K, D)

        # Pairwise squared distances via broadcasting: (N, K)
        sq_dists = np.sum(
            (embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        return float(np.sum(np.min(sq_dists, axis=1)))

    # -- archive add() tracking -----------------------------------------------

    @contextmanager
    def _track_add_status(self):
        """Context manager that monkey-patches ``archive.add`` to count
        new and improved elites, then restores the original method."""
        counts = {"new": 0, "improved": 0}
        original_add = self.archive.add

        def tracked_add(*args, **kwargs):
            res = original_add(*args, **kwargs)

            statuses = None
            if isinstance(res, tuple):
                statuses = res[0]
            elif hasattr(res, "status"):
                statuses = res.status
            elif isinstance(res, dict) and "status" in res:
                statuses = res["status"]

            if statuses is not None:
                arr = np.asarray(statuses)
                counts["new"] += int(np.sum(arr == AddStatus.NEW))
                counts["improved"] += int(
                    np.sum(arr == AddStatus.IMPROVE_EXISTING))

            return res

        self.archive.add = tracked_add
        try:
            yield counts
        finally:
            if "add" in self.archive.__dict__:
                del self.archive.__dict__["add"]

    # -- main loop ------------------------------------------------------------

    def run(self, total_iters, start_iter=1):
        """Execute the ask → evaluate → tell loop.

        Returns ``(global_best_score, global_best_id, stats)``.
        """
        for i in range(start_iter, total_iters):

            sols = self.scheduler.ask()
            sol_dicts = [array_to_solution(sol) for sol in sols]

            futs = [self.client.submit(_eval_on_worker, self.evaluator_future, sol)
                    for sol in sol_dicts]
            gathered = [f.result() for f in futs]

            obj_list, clean_solutions = [], []
            for (sol_id, ok, msg, score, measures), sol_dict in zip(gathered, sol_dicts):

                if not ok:
                    print(f"Warning: clamping bad score for ID={sol_id} ({msg})")
                    score = INVALID_SCORE
                else:
                    print(f"Solution ID={sol_id} evaluated with score={score:.2f}")
                    if score > self.global_best_score:
                        self.global_best_score = score
                        self.global_best_id = sol_id

                self._evaluation_buffer.record(sol_id, sol_dict, measures, score, ok)
                clean_solutions.append((score, measures))
                obj_list.append(score)

            obj_batch, measures_batch = zip(*clean_solutions)

            # ── Pre-tell snapshot for per-bucket tracking ──
            pre_data = self.archive.data()
            pre_obj_by_idx = {}
            if len(pre_data["objective"]) > 0:
                for _idx, _obj in zip(pre_data["index"], pre_data["objective"]):
                    pre_obj_by_idx[int(_idx)] = float(_obj)

            # ── Tell with add-status tracking ──
            with self._track_add_status() as counts:
                self.scheduler.tell(list(obj_batch), list(measures_batch))

            new_elites_count = counts["new"]
            sub_elites_count = counts["improved"]

            # ── Logging ──
            batch_best = max(obj_list) if obj_list else INVALID_SCORE
            print(f"Iteration {i} ended. Best in batch = {batch_best:.2f}")
            print(
                f"Global best so far: {self.global_best_score:.2f} (ID={self.global_best_id})")
            print(f"Archive Updates: {new_elites_count} new elites inserted, "
                  f"{sub_elites_count} elites substituted.")

            data_archive = self.archive.data()
            arch_obj = data_archive["objective"]
            valid = arch_obj != INVALID_SCORE
            mean_val = np.mean(arch_obj[valid]) if np.any(valid) else 0.0
            best_val = np.max(arch_obj[valid]) if np.any(valid) else 0.0
            elites = self.archive.stats.num_elites
            print(
                f"Archive size={elites}, mean={mean_val:.2f}, best={best_val:.2f}")

            # ── Per-bucket diff ──
            iter_new_indices = []
            iter_sub_indices = []
            if len(data_archive["objective"]) > 0:
                for _idx, _obj in zip(data_archive["index"], data_archive["objective"]):
                    idx_int = int(_idx)
                    if idx_int not in pre_obj_by_idx:
                        iter_new_indices.append(idx_int)
                        self._bucket_order.append(idx_int)
                    elif float(_obj) != pre_obj_by_idx[idx_int]:
                        iter_sub_indices.append(idx_int)
                        self._sub_counts[idx_int] = self._sub_counts.get(idx_int, 0) + 1

            # ── WSS ──
            if self.centroids is not None:
                # CVT: fixed centroids supplied at construction time
                wss_centroids = self.centroids
            else:
                # NS: use the inserted elites currently in the archive
                arch_measures = data_archive["measures"]
                wss_centroids = arch_measures if len(arch_measures) > 0 else None

            wss = self._compute_wss(wss_centroids) if wss_centroids is not None else float("nan")
            print(f"WSS = {wss:.4f}" if not np.isnan(wss) else "WSS = nan (no centroids yet)")

            self.stats.append({
                "iteration": i,
                "Archive size": elites,
                "iteration_best": batch_best,
                "global_best_score": self.global_best_score,
                "new_elites": new_elites_count,
                "substituted_elites": sub_elites_count,
                "new_bucket_indices": iter_new_indices,
                "substituted_bucket_indices": iter_sub_indices,
                "wss": wss,
            })

            # ── Checkpoint ──
            if i % CHECKPOINT_EVERY == 0:
                ckpt_name = f"{self.checkpoint_dir}checkpoint_{i:04d}.pkl"
                with open(ckpt_name, "wb") as f:
                    pickle.dump({
                        "scheduler": self.scheduler,
                        "iteration": i,
                        "global_best_score": self.global_best_score,
                        "global_best_id": self.global_best_id,
                    }, f)
                print(f"[Checkpoint] Saved {ckpt_name}")

                with open(self.stats_path, "wb") as f:
                    pickle.dump(self.stats, f)

                self._evaluation_buffer.save()

            self._visualizer.plot_heatmap(i)
            self._visualizer.plot_grid()

        # Final save
        self._evaluation_buffer.save()

        return self.global_best_score, self.global_best_id, self.stats

    @property
    def visualizer(self):
        """Access the ``ArchiveVisualizer`` for post-run plots and export."""
        return self._visualizer