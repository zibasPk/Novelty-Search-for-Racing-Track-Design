# qd_runner.py
# Shared infrastructure for Quality-Diversity search loops.
# Both novelty_search.ipynb and CVT_mapelite.ipynb delegate to these classes.

from mapelite.utils import array_to_solution
from mapelite.config import (
    BASE_URL,
    BATCH_SIZE,
    BUFFER_DIR,
    CHECKPOINT_EVERY,
    INVALID_SCORE,
    DEFAULT_START_ITER,
    HEATMAP_DIR,
    EMBEDDING_MODEL_PATH,
    PRECOMPILED_EMBEDDINGS_PATH,
    RunMode
)
from mapelite.evaluator import EvaluatorMetrics
from dask.distributed import Client, LocalCluster
from ribs.archives import AddStatus
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pickle
import datetime
import json
import joblib
import umap
import requests
from mapelite.logging_config import get_logger
from contextlib import contextmanager
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

log = get_logger(__name__)


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
            log.info("Buffer resumed", count=len(
                self.entries), path=self.buffer_path)
        else:
            log.info("Buffer empty — starting fresh", path=self.buffer_path)

    def save(self):
        """Write the full buffer to disk."""
        payload = {
            "total": len(self.entries),
            "timestamp": datetime.datetime.now().isoformat(),
            "tracks": self.entries,
        }
        with open(self.buffer_path, "w") as f:
            json.dump(payload, f)
        log.info("Buffer saved", count=len(
            self.entries), path=self.buffer_path)

    # -- recording ------------------------------------------------------------

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
        entry = {
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

    def __init__(self, archive, stats, heatmap_dir, gridplot_dir, grid_state=None, seed=None, precomp_embeddings_path=PRECOMPILED_EMBEDDINGS_PATH):
        self.archive = archive
        self.stats = stats
        self.heatmap_dir = heatmap_dir
        self.gridplot_dir = gridplot_dir
        self.seed = seed
        self._track_cache: dict = {}
        
        embeddings = np.load(precomp_embeddings_path)["embeddings"]
        umap_m = umap.UMAP(n_components=2, random_state=seed)
        
        self._umap_model = umap_m.fit(embeddings)
        self._precomp_umaps = umap_m.transform(embeddings)
        
        self._grid_state = grid_state if grid_state is not None else []
        self.prev_iteration_data = None

    @property
    def grid_state(self):
        return self._grid_state
    
    # -- track outline helper -------------------------------------------------
    def plot_grid(self, iteration_idx=None, substitutions=None, max_cols=15, max_sub_color=20,
              max_rows=None, save_dir=None):
        """Render archive buckets as a 2D grid colored by substitution count.

        Parameters
        ----------
        iteration_idx : int, optional
            Current iteration number, used only for the plot title.
        substitutions : list of (old_sol_dict, new_sol_dict), optional
            Substitution pairs returned by ``_track_add_status``.
        max_cols : int
            Fixed number of columns in the grid.
        max_sub_color : int
            Fixed upper bound for the colorbar (substitution count).
        max_rows : int, optional
            Fixed number of rows.  ``None`` → derived from current grid state size.
        save_dir : str, optional
            Directory where the figure should be saved.  Falls back to
            ``self.gridplot_dir`` if *None*.
        """
        if save_dir is None:
            save_dir = self.gridplot_dir

        if substitutions is None:
            substitutions = []
            

        archive = self.archive
        elites_dicts = [array_to_solution(sol) for sol in archive.data("solution")]
        id_to_fitness = dict(zip((sol_dict["id"] for sol_dict in elites_dicts), archive.data("objective")))
        
        in_grid = lambda id: any(item["elite"]["id"] == id for item in self._grid_state)
        is_substitution = lambda id: any(item[1]["id"] == id for item in substitutions)
        

        # Add new elites that aren't already tracked and aren't replacing an existing slot
        for elite in elites_dicts:
            if not in_grid(elite["id"]) and not is_substitution(elite["id"]):
                self._grid_state.append({
                    "elite": elite,
                    "sub_count": 0,
                    "new": True,
                    "fitness": id_to_fitness.get(elite["id"], np.nan)
                })
                
        # Apply substitutions, in-place preserving grid position when possible
        subs_by_id = defaultdict(list)
        for prev_sol, new_sol in substitutions:
            if new_sol["id"] == 3.2919232767901017:
                print(f"Found solution with ID {new_sol['id']} in substitution check")
            subs_by_id[prev_sol["id"]].append(new_sol)

        items_to_add = []
        for item in self._grid_state:
            new_sols = subs_by_id.pop(item["elite"]["id"], None)
            if new_sols is None:
                continue

            item_snapshot = item.copy()  # snapshot before mutating, for appended copies

            # First substitution: update in place
            item["elite"]   = new_sols[0]
            item["sub_count"] += 1
            item["new"]     = True
            item["fitness"] = id_to_fitness.get(new_sols[0]["id"], np.nan)

            # Remaining substitutions: append copies of the original slot
            for new_sol in new_sols[1:]:
                new_item = item_snapshot.copy()
                new_item["elite"]     = new_sol
                new_item["sub_count"] += 1
                new_item["new"]       = True
                new_item["fitness"]   = id_to_fitness.get(new_sol["id"], np.nan)
                items_to_add.append(new_item)

        self._grid_state.extend(items_to_add)  
        

        if len(self._grid_state) == 0:
            return

        # check if all elites in gridstate are still in the archive; if not, log a warning
        archive_ids = set(sol_dict["id"] for sol_dict in elites_dicts)
        for item in self._grid_state:
            if item["elite"]["id"] not in archive_ids:
                log.warning("Elite in grid state no longer in archive",
                            elite_id=item["elite"]["id"])
        
        # check if all elites in archive are in gridstate; if not, log a warning (but don't add them to the gridstate since they may have been substituted in later batches)    
        for elite in elites_dicts:
            if not in_grid(elite["id"]):
                log.warning("Elite in archive not found in grid state",
                            elite_id=elite["id"])
        #check if all substitutions are valid (new solutions are in the archive); if not, log a warning
        for _, new_sol in substitutions:
            if new_sol["id"] not in archive_ids:
                log.warning("Substituted solution not found in archive",
                            new_id=new_sol["id"])
            
  
        archive_ids = [sol_dict["id"] for sol_dict in elites_dicts]
        substitutions_info = [(prev["id"], new["id"]) for prev, new in substitutions]
        grid_state_ids_sub_counts = [(item["elite"]["id"], item["sub_count"]) for item in self._grid_state]
        
        prev_elite_ids = []
        prev_substitutions_info = []
        prev_grid_state_ids_sub_counts = []
        if self.prev_iteration_data is not None:
            prev_elite_ids = [sol_dict["id"] for sol_dict in self.prev_iteration_data["elites"]]
            prev_substitutions_info = [(prev["id"], new["id"]) for prev, new in self.prev_iteration_data["substitutions"]]
            prev_grid_state_ids_sub_counts = [(item["elite"]["id"], item["sub_count"]) for item in self.prev_iteration_data["grid_state"]]
        
        log.debug("Debug info for grid plot",
                    elite_ids=archive_ids,
                    substitutions=substitutions_info,
                    grid_state=grid_state_ids_sub_counts,
                    prev_elite_ids=prev_elite_ids,
                    prev_substitutions=prev_substitutions_info,
                    prev_grid_state=prev_grid_state_ids_sub_counts
                    )
        
        self.prev_iteration_data = {
            "elites": elites_dicts,
            "id_to_fitness": id_to_fitness,
            "substitutions": substitutions,
            "grid_state": self._grid_state.copy()
        }
        
       
        
        # ── Fixed grid dimensions ──
        cols = max_cols
        total = len(self._grid_state)
        if max_rows is None:
            max_rows = max(1, int(np.ceil(total / cols)))

        # Build grid of substitution counts; NaN = unfilled cell
        grid = np.full((max_rows, cols), np.nan)
        for pos, item in enumerate(self._grid_state):
            r, c = divmod(pos, cols)
            if r < max_rows:
                grid[r, c] = item["sub_count"]

        cmap = LinearSegmentedColormap.from_list("sub", ["white", "#ffff00", "red"])
        cmap.set_bad("lightgray")

        fig_w = max(6, cols * 0.6)
        fig_h = max(4, max_rows * 0.6)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=max_sub_color,
                    aspect="equal", interpolation="nearest",
                    origin="upper",
                    extent=(-0.5, cols - 0.5, max_rows - 0.5, -0.5))

        plt.colorbar(im, ax=ax, label="Substitution Count", fraction=0.046, pad=0.04)

        # ── Draw track outlines and annotations ──
        for pos, item in enumerate(self._grid_state):
            r, c = divmod(pos, cols)
            if r >= max_rows:
                break

            elite = item["elite"]
            count = item["sub_count"]
            is_new = item["new"]

            # Track outline
            outline = self._get_track_outline(elite)
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
                    track_color = "blue" if is_new else "black"
                    track_lw = 0.9 if is_new else 0.4
                    ax.plot(cell_xs, cell_ys, color=track_color,
                            linewidth=track_lw, alpha=0.55, zorder=2)

            # Substitution count (top-left)
            if count > 0:
                ax.text(c - 0.42, r - 0.40, str(count),
                        ha="left", va="top", fontsize=5, zorder=3,
                        color="black" if count < max_sub_color * 0.6 else "white")

            # Fitness score (bottom-right)
            fit_val = item.get("fitness")
            if fit_val is not None:
                if np.isfinite(fit_val) and fit_val != INVALID_SCORE:
                    ax.text(c + 0.44, r + 0.44, f"{fit_val:.1f}",
                            ha="right", va="bottom", fontsize=4, zorder=3, color="blue")
                elif fit_val == INVALID_SCORE:
                    ax.text(c + 0.44, r + 0.44, "X",
                            ha="right", va="bottom", fontsize=5, zorder=3,
                            color="red", fontweight="bold")
            else:
                log.warning("Elite missing fitness for grid annotation", elite_id=elite["id"])

        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, max_rows, 1), minor=True)
        ax.grid(which="minor", color="gray", linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False,
                    labelbottom=False, labelleft=False)
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(max_rows - 0.5, -0.5)

        it_label = iteration_idx if iteration_idx is not None else len(self.stats) - 1
        ax.set_title(f"Archive Grid — Iteration {it_label}  "
                    f"({total} / {max_rows * cols} buckets filled)")

        plt.tight_layout()
        
        # save grid_state_ids_sub_counts json for this iteration
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"archive_grid_iter_{it_label:04d}.png"),
                        dpi=200, bbox_inches="tight")
            plt.close(fig)
            
            grid_save_path = os.path.join(save_dir, f"grid_state_ids_iter_{it_label:04d}.json")
            with open(grid_save_path, "w") as f:
                json.dump(grid_state_ids_sub_counts, f, indent=2)
        else:
            plt.show()
            plt.close(fig)

        # Mark all entries as no longer new for the next call
        for item in self._grid_state:
            item["new"] = False
        
        
    
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
        except Exception as exc:
            log.debug("Track reconstruct failed",
                      sol_id=sol_id, error=str(exc))

        # Cache only successful reconstructions so transient failures can retry
        # on the next plotting call instead of staying permanently empty.
        if result is not None:
            self._track_cache[sol_id] = result
        return result
    
    def plot_heatmap(self, iteration, save_dir=None):
        """2D UMAP compression of the archive's behavioural space, colored by fitness."""
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
        cmap.set_under("deeppink")
        cmap.set_over("red")
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        # ── Background: full precomputed UMAP landscape ──────────────────────────
        ax.scatter(
            self._precomp_umaps[:, 0],
            self._precomp_umaps[:, 1],
            c="lightgray",
            s=3,
            alpha=0.30,
            linewidths=0,
            zorder=1,
            label="All candidates",
        )
        # ── Foreground: current archive elites ───────────────────────────────────
        sc = ax.scatter(
            umap_compression[:, 0], umap_compression[:, 1],
            c=fitnesses, cmap=cmap, s=10, vmin=MIN_FIT, vmax=MAX_FIT,
            zorder=2,
        )
        plt.colorbar(sc, ax=ax, label="Fitness", extend="both")

        # ── Bounds: fit to _precomp_umaps, expanding if archive elites exceed them ─
        x_min = min(self._precomp_umaps[:, 0].min(), umap_compression[:, 0].min())
        x_max = max(self._precomp_umaps[:, 0].max(), umap_compression[:, 0].max())
        y_min = min(self._precomp_umaps[:, 1].min(), umap_compression[:, 1].min())
        y_max = max(self._precomp_umaps[:, 1].max(), umap_compression[:, 1].max())
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # ── Title with iteration number ───────────────────────────────────────────
        ax.set_title(f"Archive Heatmap — Iteration {iteration}")

        plt.tight_layout()
        fig.savefig(os.path.join(
            save_dir, f"archive_heatmap_iter_{iteration:04d}.png"))
        plt.close(fig)
        np.savez_compressed(os.path.join(save_dir, f"archive_data_iter_{iteration:04d}.npz"),
                            umap=umap_compression, fitness=fitnesses, solutions=solution_dicts)

    # -- stats plot -----------------------------------------------------------

    def plot_stats(self, title="QD Run Statistics"):
        """Modular run-statistics plot.

        Each subplot is driven by an entry in ``PANELS`` below.  To add a new
        graph, append one more dict to the list — no other code needs to
        change.  If a key is absent from the stats (or all-NaN), the panel is
        rendered as an empty placeholder so the layout stays consistent.

        Panel ``type`` values
        ---------------------
        ``"line"``        – simple line plot; requires ``key``, ``color``.
        ``"bar"``         – bar chart;         requires ``key``, ``color``.
        ``"multi_line"``  – overlaid lines;    requires ``series`` list of
                            ``{key, label, color, alpha?, linewidth?,
                            clean_invalid?}``.
        ``"cumulative"``  – cumulative new + substituted elites (no extra keys).
        ``"wss"``         – line plot with optional ``initial_WSS`` reference;
                            requires ``key``, ``color``.
        """
        stats = self.stats
        if not stats:
            print("No stats to plot.")
            return

        iterations = [s["iteration"] for s in stats]
        initial_WSS = stats[0].get("initial_WSS")
        bar_width = max(1, len(iterations) // 200)

        def get_series(key):
            return [s.get(key, float("nan")) for s in stats]

        def is_empty(values):
            for v in values:
                try:
                    if not np.isnan(float(v)):
                        return False
                except (TypeError, ValueError):
                    return False
            return True

        # ── Panel definitions (extend this list to add new graphs) ────────────
        PANELS = [
            {
                "title": "Archive Growth", "ylabel": "Archive Size",
                "type": "line", "key": "Archive size", "color": "tab:blue",
            },
            {
                "title": "Fitness Progress", "ylabel": "Fitness Score",
                "type": "multi_line",
                "series": [
                    {"key": "iteration_best",   "label": "Iteration Best",
                     "color": "tab:orange", "alpha": 0.6, "linewidth": 1,
                     "clean_invalid": True},
                    {"key": "global_best_score", "label": "Global Best",
                     "color": "tab:red", "linewidth": 2},
                ],
            },
            {
                "title": "New Elites per Iteration", "ylabel": "Count",
                "type": "bar", "key": "new_elites", "color": "tab:red",
            },
            {
                "title": "Substituted Elites per Iteration", "ylabel": "Count",
                "type": "bar", "key": "substituted_elites", "color": "tab:blue",
            },
            {
                "title": "Cumulative Elite Insertions",
                "ylabel": "Cumulative Count",
                "type": "cumulative",
            },
            {
                "title": "Within-Cluster Sum of Squares (WSS) — normalized by evaluated tracks",
                "ylabel": "Mean WSS/track",
                "type": "wss", "key": "wss", "color": "tab:brown",
            },
            {
                "title": "QD-Score", "ylabel": "QD-Score",
                "type": "line", "key": "qd_score", "color": "tab:green",
            },
            {
                "title": "Archive Acceptance Rate", "ylabel": "Acceptance Rate",
                "type": "line", "key": "acceptance_rate", "color": "tab:orange",
            },
            {
                "title": "Mean Pairwise Distance",
                "ylabel": "Mean Distance",
                "type": "line", "key": "mean_pairwise_dist", "color": "tab:purple",
            },
            {
                "title": "High-Quality Coverage",
                "ylabel": "Count",
                "type": "line", "key": "high_quality_coverage", "color": "darkred",
            },
            {
                "title": "Mean k-NN Novelty Score (NS only)",
                "ylabel": "Mean k-NN Distance",
                "type": "line", "key": "mean_knn_novelty", "color": "tab:cyan",
            },
            {
                "title": "Fitness\u2013Novelty Correlation (NS only)",
                "ylabel": "Pearson r",
                "type": "line", "key": "fitness_novelty_corr", "color": "tab:pink",
            },
        ]

        n_panels = len(PANELS)
        fig, axes = plt.subplots(n_panels, 1,
                                 figsize=(14, n_panels * 2.5), sharex=True)
        if n_panels == 1:
            axes = [axes]
        fig.suptitle(f"{title} — Run Statistics",
                     fontsize=16, fontweight="bold")

        for ax, p in zip(axes, PANELS):
            ptype = p["type"]

            if ptype == "line":
                values = get_series(p["key"])
                if is_empty(values):
                    ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                            transform=ax.transAxes, color="gray", fontsize=10)
                else:
                    ax.plot(iterations, values,
                            color=p["color"], linewidth=1.5)

            elif ptype == "bar":
                values = get_series(p["key"])
                if is_empty(values):
                    ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                            transform=ax.transAxes, color="gray", fontsize=10)
                else:
                    ax.bar(iterations, values, width=bar_width,
                           color=p["color"], alpha=0.8)

            elif ptype == "multi_line":
                for s_cfg in p["series"]:
                    values = get_series(s_cfg["key"])
                    if s_cfg.get("clean_invalid"):
                        values = [
                            v if v != INVALID_SCORE else np.nan for v in values]
                    ax.plot(iterations, values,
                            label=s_cfg.get("label"),
                            color=s_cfg.get("color", "black"),
                            alpha=s_cfg.get("alpha", 1.0),
                            linewidth=s_cfg.get("linewidth", 1.5))
                ax.legend()

            elif ptype == "cumulative":
                new_e = get_series("new_elites")
                sub_e = get_series("substituted_elites")
                cum_new = np.cumsum(new_e)
                cum_sub = np.cumsum(sub_e)
                ax.plot(iterations, cum_new, label="Cumulative New",
                        color="tab:green", linewidth=1.5)
                ax.plot(iterations, cum_sub, label="Cumulative Substituted",
                        color="tab:purple", linewidth=1.5)
                ax.plot(iterations, cum_new + cum_sub, label="Cumulative Total",
                        color="tab:blue", linewidth=2, linestyle="--")
                ax.legend()

            elif ptype == "wss":
                values = get_series(p["key"])
                if is_empty(values):
                    ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                            transform=ax.transAxes, color="gray", fontsize=10)
                else:
                    ax.plot(iterations, values,
                            color=p["color"], linewidth=1.5)
                if initial_WSS is not None:
                    ax.axhline(initial_WSS, color="red", linewidth=1.5,
                               linestyle="--",
                               label=f"Training mean WSS/track ({initial_WSS:.2f})")
                    ax.legend()

            ax.set_ylabel(p.get("ylabel", ""))
            ax.set_title(p["title"])
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Iteration")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

        # ── Summary printout ─────────────────────────────────────────────────
        archive_sizes = get_series("Archive size")
        global_best = get_series("global_best_score")
        new_elites = get_series("new_elites")
        sub_elites = get_series("substituted_elites")
        log.info(
            "Run summary",
            title=title,
            total_iterations=len(stats),
            final_archive_size=archive_sizes[-1],
            global_best_fitness=f"{global_best[-1]:.4f}",
            total_new_elites=sum(new_elites),
            total_substituted=sum(sub_elites),
            avg_new_per_iter=f"{np.mean(new_elites):.2f}",
            avg_sub_per_iter=f"{np.mean(sub_elites):.2f}",
        )

    # -- elite export ---------------------------------------------------------

    def export_elites(self, output_path, algorithm_label, seed):
        """Save all valid elites to a JSON file for reconstruction & visualization."""
        
        global_best_score = self.stats[-1]["global_best_score"]
        global_best_id = self.stats[-1]["global_best_id"]

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
            log.warning("Skipped elites with invalid fitness",
                        count=skipped_invalid)

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

        log.info(
            "Elites exported",
            count=len(elites_list),
            path=output_path,
            best_fitness=f"{elites_list[0]['fitness']:.4f}",
            best_id=elites_list[0]["id"],
            worst_fitness=f"{elites_list[-1]['fitness']:.4f}",
            file_kb=f"{os.path.getsize(output_path) / 1024:.1f}",
        )


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
        
        # Evaluation buffer & track cache
        self._evaluation_buffer = EvaluationBuffer(self.buffer_path)
        self._visualizer = ArchiveVisualizer(
            archive, self.stats, heatmap_dir, gridplot_dir, seed=seed, grid_state = grid_state)
        
        self._run_mode = RunMode.CVT if centroids is not None else RunMode.NS
        
    @classmethod
    def load_state(cls, state, client, evaluator_future, checkpoint_dir, heatmap_dir, gridplot_dir, buffer_path, seed):
        instance = cls(
            scheduler = state["scheduler"],
            archive = state["archive"],
            start_iter = state["start_iter"],
            stats = state["stats"],
            evaluator_future = evaluator_future,
            client = client,
            checkpoint_dir = checkpoint_dir,
            heatmap_dir = heatmap_dir,
            gridplot_dir = gridplot_dir,
            buffer_path = buffer_path,
            seed = seed,
            centroids = getattr(state["archive"], "centroids", None),
            initial_WSS = state["stats"][0].get("initial_WSS", None),
            grid_state = state["stats"][-1].get("grid_state", None)
        )
        
        return instance


    @staticmethod
    def setup_dask(batch_size=BATCH_SIZE, model_path=None):
        """Create a Dask LocalCluster and scatter the evaluator to all workers.

        Returns ``(client, cluster, evaluator_future)``.
        """
        log.debug("Setting up Dask LocalCluster", n_workers=batch_size)
        cluster = LocalCluster(
            processes=True, n_workers=batch_size, threads_per_worker=1)
        client = Client(cluster)
        log.debug("Dask cluster ready", dashboard=client.dashboard_link)

        evaluator = EvaluatorMetrics.load_pretrained(model_path)
        evaluator_future = client.scatter(evaluator, broadcast=True)
        log.debug("Evaluator scattered to Dask workers", n_workers=batch_size)

        return client, cluster, evaluator_future

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
        AddStatus_counts = {AddStatus.NEW: 0, AddStatus.IMPROVE_EXISTING: 0}
        counts = {"new": 0, "improved": 0}
        substitutions = []  # (old_id, new_id)
        new_insertions = []  # (new_id)
        original_add = self.archive.add

        def tracked_add(*args, **kwargs):
            solutions_batch, measures_batch = kwargs.get("solution"), kwargs.get("measures")
            solutions_batch_dicts = [array_to_solution(sol) for sol in solutions_batch]
            emptystart = self.archive.stats.num_elites == 0
            
            self_pre_archive_dicts_set = {array_to_solution(sol)["id"]: array_to_solution(sol) for sol in self.archive.data("solution")}
            
            if not emptystart:
                _, pre_neighbours = self.archive.retrieve(measures_batch)
                pre_neighbours_sols = [array_to_solution(sol) for sol in pre_neighbours["solution"]]
            
            res = original_add(*args, **kwargs)
            
            statuses = res["status"]

            AddStatus_counts[AddStatus.NEW] += int(np.sum(statuses == AddStatus.NEW))
            AddStatus_counts[AddStatus.IMPROVE_EXISTING] += int(np.sum(statuses == AddStatus.IMPROVE_EXISTING))

            if not emptystart:
                _, post_neighbours = self.archive.retrieve(measures_batch)
                post_neighbours_sols = [array_to_solution(sol) for sol in post_neighbours["solution"]]
                
                # check for each old neighbour what the new closest neighbour is after the batch add
                _, pre_neighbor_new_nearest  = self.archive.retrieve(pre_neighbours["measures"])
                pre_neighbor_new_nearest_sols = [array_to_solution(sol) for sol in pre_neighbor_new_nearest["solution"]]
                
                for idx, (batch_sol, challenged_new) in enumerate(zip(solutions_batch_dicts, post_neighbours_sols)):
                    if batch_sol["id"] == challenged_new["id"]:
                        # elite newly inserted or improved
                        if pre_neighbours_sols[idx]["id"] == pre_neighbor_new_nearest_sols[idx]["id"]:
                            # elite improved but didn't substitute its previous neighbour as the closest one, so it must have been a new insertion rather than a substitution
                            new_insertions.append(batch_sol["id"])
                        else:
                            # elite improved and substituted its previous neighbour
                            substitutions.append((pre_neighbours_sols[idx], batch_sol))
                    else:
                        # elite didn't win competition
                        continue
            
            if len(substitutions) > 0:
                # now check for any self substitutions among the batch solutions
                subbing_ids = [sub[1]["id"] for sub in substitutions]
                
                subbing_measures = [m for m, s in zip(measures_batch, solutions_batch_dicts) if s["id"] in subbing_ids]
            
                _, neighbours = self.archive.retrieve(subbing_measures)
                neighbours_ids = [array_to_solution(sol)["id"] for sol in neighbours["solution"]]
                
                # if the neighbour isn't itself, then it must have been substituted by another batch solution
                # NOTE: in the proximity archive it is possible that two solutions substitute the same element, but are both kept 
                for subber_id, neighbour_id in zip(subbing_ids, neighbours_ids):
                   
                    if subber_id != neighbour_id:
                        # remove from substitutions
                        substitutions[:] = [s for s in substitutions if s[1]["id"] != subber_id]

            # update counts 
            counts["new"] = len(new_insertions)
            counts["improved"] = len(substitutions)     
        
            # check wheter its consistent with archive contents
            archive_data = self.archive.data("solution")
            archive_ids = set(array_to_solution(sol)["id"] for sol in archive_data)
            for sub in substitutions:
                if sub[0]["id"] in archive_ids:
                    log.warning("Substituted elite still in archive",
                                old_id=sub[0]["id"], new_id=sub[1]["id"])
                if sub[1]["id"] not in archive_ids:
                    log.warning("Substituting elite not in archive",
                                new_id=sub[1]["id"])
            
            for sol in new_insertions:
                if sol not in archive_ids:
                    log.warning("New elite not in archive",
                                new_id=sol["id"])
            
            post_archive_dicts_set = {array_to_solution(sol)["id"]: array_to_solution(sol) for sol in self.archive.data("solution")}
            
            # get new elites from sets
            new_elites_from_sets = post_archive_dicts_set.keys() - self_pre_archive_dicts_set.keys()
            # get removed elites from sets
            removed_elites_from_sets = self_pre_archive_dicts_set.keys() - post_archive_dicts_set.keys()
            
            # fore each new elite check if it is in the tracked new insertions or substitutions, otherwise log a warning
            for new_elite_id in new_elites_from_sets:
                if new_elite_id not in [s[1]["id"] for s in substitutions] and new_elite_id not in new_insertions:
                    log.warning("New elite not tracked as new or improved",
                                new_id=new_elite_id)
                    
            # for each removed elite check if it is in the tracked substitutions, otherwise log a warning
            for removed_elite_id in removed_elites_from_sets:
                if removed_elite_id not in [s[0]["id"] for s in substitutions]:
                    log.warning("Removed elite not tracked as substituted",
                                removed_id=removed_elite_id)
            

            return res

        self.archive.add = tracked_add
        try:
            yield counts, substitutions
        finally:
            if "add" in self.archive.__dict__:
                del self.archive.__dict__["add"]

    # -- main loop ------------------------------------------------------------

    def run(self, total_iters, start_iter=None):
        """Execute the ask → evaluate → tell loop.

        Returns ``(global_best_score, global_best_id, stats)``.
        """
        if start_iter is None:
            start_iter = self.start_iter

        for i in range(start_iter, total_iters):

            sols = self.scheduler.ask()
            sol_dicts = [array_to_solution(sol) for sol in sols]

            futs = [self.client.submit(_eval_on_worker, self.evaluator_future, sol)
                    for sol in sol_dicts]
            gathered = [f.result() for f in futs]

            score_list, clean_solutions = [], []
            # Measure is the embedding returned by the evaluator
            for (sol_id, ok, msg, objective_score, measure, phenotype_data), sol_dict in zip(gathered, sol_dicts):

                if sol_id == 22.474385782410586:
                    print(f"Found solution with ID {sol_id}")

                if not ok:
                    log.warning("Clamping bad score",
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
            with self._track_add_status() as (counts, substitutions):
                self.scheduler.tell(list(score_batch), list(measures_batch))


            # Metrics and Visualizations:
            new_elites_count = counts["new"]
            sub_elites_count = counts["improved"]
            batch_best = max(score_list) if score_list else INVALID_SCORE

            log.info(
                "Iteration complete",
                iteration=i,
                batch_best=f"{batch_best:.2f}",
                global_best=f"{self.global_best_score:.2f}",
                global_best_id=self.global_best_id,
                new_elites=new_elites_count,
                substituted=sub_elites_count,
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
            high_quality_cov = self._compute_high_quality_coverage(arch_scores)
            
            mean_knn_novelty = None
            fitness_novelty_corr = None
            wss = None
            
            match self._run_mode:
                case RunMode.CVT:
                    wss = self._compute_wss(self.centroids) 
                case RunMode.NS:
                    mean_knn_novelty = self._compute_mean_knn_novelty(measures)
                    fitness_novelty_corr = self._compute_fitness_novelty_corr(measures, arch_scores)
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

            # its important to make sure it runs at least the first iteration
            if i == 0 or new_elites_count > 0 or sub_elites_count > 0:
                self._visualizer.plot_heatmap(i)
                self._visualizer.plot_grid(i, substitutions)
            
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
