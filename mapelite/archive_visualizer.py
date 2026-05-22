# archive_visualizer.py
# Visualization utilities for QD archive inspection.

from mapelite.utils import array_to_solution
from mapelite.config import (
    BASE_URL,
    INVALID_SCORE,
    PRECOMPILED_EMBEDDINGS_PATH,
)
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import datetime
import umap
import requests
from mapelite.logging_config import get_logger

log = get_logger(__name__)


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

        # Apply substitutions, in-place preserving grid position
        for prev_sol, new_sol in substitutions:
            for item in self._grid_state:
                if item["elite"]["id"] == prev_sol["id"]:
                    item["elite"] = new_sol
                    item["sub_count"] += 1
                    item["fitness"] = id_to_fitness.get(new_sol["id"], np.nan)
                    item["new"] = True
                    break
            else:
                log.error("Substitution target not found in grid state",
                            target_id=prev_sol["id"], new_id=new_sol["id"])

        if len(self._grid_state) == 0:
            return

        # check if all elites in gridstate are still in the archive;
        archive_ids = set(sol_dict["id"] for sol_dict in elites_dicts)
        for item in self._grid_state:
            if item["elite"]["id"] not in archive_ids:
                log.error("Elite in grid state no longer in archive",
                            elite_id=item["elite"]["id"])

        # check if all elites in archive are in gridstate;
        for elite in elites_dicts:
            if not in_grid(elite["id"]):
                log.error("Elite in archive not found in grid state",
                            elite_id=elite["id"])

        grid_state_ids_sub_counts = [(item["elite"]["id"], item["sub_count"]) for item in self._grid_state]

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
            r_json = resp.json()
            if not resp.ok:
                raise Exception(f"API error {resp.status_code}: {r_json.get('error', resp.text)}")

            track = r_json.get("track", [])
            if track:
                xs = np.array([p["x"] for p in track], dtype=float)
                ys = np.array([p["y"] for p in track], dtype=float)
                result = (xs, ys)
        except Exception as exc:
            log.debug("Track reconstruct failed:", error=str(exc), sol_id=sol_id)

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

    def plot_stats(self, title="QD Run Statistics", stats_dir=None):
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
                "title": "Fitness–Novelty Correlation (NS only)",
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

        # Save plot image to file
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
            fig.savefig(os.path.join(stats_dir, "run_stats.png"), dpi=200)
        plt.close(fig)

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
