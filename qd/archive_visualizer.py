# archive_visualizer.py
# Visualization utilities for QD archive inspection.

from qd.utils import array_to_solution
from qd.config import (
    BASE_URL,
    INVALID_SCORE,
    RETRAIN_EVERY
)
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import re
import json
import datetime
from io import BytesIO
import requests
from qd.logging_config import get_logger

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
    images_dir : str, optional
        Directory to save exported elite track images.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, archive, stats, images_dir=None, seed=None):
        self.archive = archive
        self.stats = stats
        self.images_dir = images_dir
        self.seed = seed
        self._track_cache: dict = {}

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

    # Metrics used to produce per-metric colored image sets.
    # Each entry: (buffer key, human label, colormap name)
    COLORING_METRICS = [
        ("curvature_entropy",  "curvature_entropy",  "plasma"),
        ("avg_radius_mean",    "avg_radius_mean",     "viridis"),
        ("speed_entropy",      "speed_entropy",       "inferno"),
        ("total_overtakes",    "total_overtakes",     "YlOrRd"),
    ]

    @staticmethod
    def _render_fig(fig, dpi=150):
        """Save *fig* into a BytesIO buffer and return ``(png_bytes, uint8 RGBA array)``."""
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        png_bytes = buf.getvalue()
        buf.seek(0)
        img = plt.imread(buf)                        # float32 RGBA [0, 1]
        return png_bytes, (img * 255).astype(np.uint8)

    def save_elite_images(self, iteration_idx, save_dir=None, evaluation_buffer=None, save_pngs=False):
        """For each elite in the archive, render its track outline into a plain PNG and a set of colored PNGs (one per metric in COLORING_METRICS), and save all images + raw metric values into a compressed NPZ file."""
        if save_dir is None:
            save_dir = self.images_dir

        iter_dir = os.path.join(save_dir, str(iteration_idx))
        os.makedirs(iter_dir, exist_ok=True)

        elites_dicts = [array_to_solution(sol) for sol in self.archive.data("solution")]

        outlines = []
        max_span = 0.0
        for elite in elites_dicts:
            outline = self._get_track_outline(elite)
            if outline is None:
                continue
            xs, ys = outline
            outlines.append((elite, xs, ys))
            max_span = max(max_span, xs.max() - xs.min(), ys.max() - ys.min())

        if not outlines:
            log.info("No elite track outlines available to save", iteration=iteration_idx)
            return

        half_extent = max_span / 2 * 1.05

        npz_arrays = {"ids": np.array([elite["id"] for elite, _, _ in outlines])}

        # ── Plain images ──────────────────────────────────────────────────────
        if save_pngs:
            plain_dir = os.path.join(iter_dir, "plain")
            os.makedirs(plain_dir, exist_ok=True)
        plain_imgs = []
        for elite, xs, ys in outlines:
            cx = (xs.max() + xs.min()) / 2
            cy = (ys.max() + ys.min()) / 2
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.plot(xs, ys, color="crimson", linewidth=1.2, zorder=2)
            ax.set_aspect("equal")
            ax.set_xlim(cx - half_extent, cx + half_extent)
            ax.set_ylim(cy - half_extent, cy + half_extent)
            ax.set_axis_off()
            png_bytes, img_arr = self._render_fig(fig)
            if save_pngs:
                with open(os.path.join(plain_dir, f"elite_{elite['id']}.png"), "wb") as f:
                    f.write(png_bytes)
            plain_imgs.append(img_arr)
            plt.close(fig)
        npz_arrays["plain"] = np.array(plain_imgs)

        # ── Colored images ────────────────────────────────────────────────────
        if evaluation_buffer is not None:
            bg_cmap = LinearSegmentedColormap.from_list("white_green", ["white", "green"])

            elite_raw = {
                elite["id"]: (evaluation_buffer.entries.get(elite["id"]) or {}).get("raw_fitness") or {}
                for elite, _, _ in outlines
            }

            for buf_key, label, _ in self.COLORING_METRICS:
                vals = {
                    elite["id"]: elite_raw[elite["id"]].get(buf_key)
                    for elite, _, _ in outlines
                    if elite_raw[elite["id"]].get(buf_key) is not None
                }
                if not vals:
                    log.debug("Skipping coloring — metric absent from buffer", metric=label)
                    continue

                v_min, v_max = min(vals.values()), max(vals.values())
                norm = plt.Normalize(vmin=v_min, vmax=v_max) if v_max > v_min else None

                if save_pngs:
                    metric_dir = os.path.join(iter_dir, f"colored_{label}")
                    os.makedirs(metric_dir, exist_ok=True)
                colored_imgs = []
                metric_raw_vals = []

                for elite, xs, ys in outlines:
                    cx = (xs.max() + xs.min()) / 2
                    cy = (ys.max() + ys.min()) / 2
                    v = vals.get(elite["id"])
                    bg_color = bg_cmap(norm(v)) if (v is not None and norm is not None) else "lightgray"
                    metric_raw_vals.append(float(v) if v is not None else np.nan)

                    fig, ax = plt.subplots(figsize=(4, 4))
                    fig.patch.set_facecolor(bg_color)
                    ax.set_facecolor(bg_color)
                    ax.plot(xs, ys, color="crimson", linewidth=1.5, zorder=2)
                    ax.set_aspect("equal")
                    ax.set_xlim(cx - half_extent, cx + half_extent)
                    ax.set_ylim(cy - half_extent, cy + half_extent)
                    ax.set_axis_off()
                    png_bytes, img_arr = self._render_fig(fig)
                    if save_pngs:
                        with open(os.path.join(metric_dir, f"elite_{elite['id']}.png"), "wb") as f:
                            f.write(png_bytes)
                    colored_imgs.append(img_arr)
                    plt.close(fig)

                npz_arrays[f"colored_{label}"] = np.array(colored_imgs)
                npz_arrays[f"metric_{label}"] = np.array(metric_raw_vals, dtype=np.float32)

        npz_path = os.path.join(iter_dir, "elite_images.npz")
        np.savez_compressed(npz_path, **npz_arrays)

        log.info("Elite track images saved", count=len(outlines),
                 total=len(elites_dicts), path=iter_dir)

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
        ``"sparse_line"`` – line+markers over only the iterations that carry a
                            value (NaN/missing skipped); requires ``key``,
                            ``color``.  Use for metrics recorded sporadically
                            (e.g. recon loss, logged only on retraining iters).
        ``"bar"``         – bar chart;         requires ``key``, ``color``.
        ``"multi_line"``  – overlaid lines;    requires ``series`` list of
                            ``{key, label, color, alpha?, linewidth?,
                            clean_invalid?}``.

        Panels are laid out two per row, each with its own x-axis labelled in
        iterations at a granularity of 150.
        """
        stats = self.stats
        if not stats:
            print("No stats to plot.")
            return

        iterations = [s["iteration"] for s in stats]
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
                     "color": "tab:orange", "alpha": 0.45, "linewidth": 1,
                     "clean_invalid": True},
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
                "title": "New vs Substituted Elites per Iteration",
                "ylabel": "Count",
                "type": "overlay_bar",
                # Drawn back-to-front: new insertions sit at the back (opaque),
                # substitutions on top with lower opacity.
                "series": [
                    {"key": "new_elites", "label": "New", "color": "tab:red",
                     "alpha": 0.8, "zorder": 1},
                    {"key": "substituted_elites", "label": "Substituted",
                     "color": "tab:blue", "alpha": 0.5, "zorder": 2},
                ],
            },
            {
                "title": "Mean Archive Fitness", "ylabel": "Mean Fitness",
                "type": "line", "key": "mean_fitness", "color": "tab:purple",
            },
            {
                "title": "Archive Acceptance Rate", "ylabel": "Acceptance Rate",
                "type": "line", "key": "acceptance_rate", "color": "tab:orange",
            },
            {
                "title": "High-Quality Coverage",
                "ylabel": "Count",
                "type": "line", "key": "high_quality_coverage", "color": "darkred",
            },
            {
                "title": "Fitness–Novelty Correlation (NS only)",
                "ylabel": "Pearson r",
                "type": "line", "key": "fitness_novelty_corr", "color": "tab:pink",
            },
            {
                "title": "Reconstruction Loss after Retraining",
                "ylabel": "Val Recon Loss",
                "type": "sparse_line", "key": "recon_loss", "color": "tab:cyan",
            },
        ]

        n_panels = len(PANELS)
        n_cols = 2
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(14, n_rows * 3), squeeze=False)
        axes_flat = axes.flatten()
        fig.suptitle(f"{title} — Run Statistics",
                     fontsize=16, fontweight="bold")

        def render_panel(ax, p):
            """Draw a single panel ``p`` onto ``ax``. Shared by the combined
            grid and the per-panel standalone PNGs so both stay in sync."""
            ptype = p["type"]

            if ptype == "line":
                values = get_series(p["key"])
                if is_empty(values):
                    ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                            transform=ax.transAxes, color="gray", fontsize=10)
                else:
                    ax.plot(iterations, values,
                            color=p["color"], linewidth=1.5)

            elif ptype == "sparse_line":
                # Plot only the iterations that actually carry a value (e.g.
                # recon loss, recorded only on retraining iterations).
                values = get_series(p["key"])
                pts = [(it, float(v)) for it, v in zip(iterations, values)
                       if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if not pts:
                    ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                            transform=ax.transAxes, color="gray", fontsize=10)
                else:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, color=p["color"], linewidth=1.5,
                            marker="o", markersize=4)

            elif ptype == "bar":
                values = get_series(p["key"])
                if is_empty(values):
                    ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                            transform=ax.transAxes, color="gray", fontsize=10)
                else:
                    ax.bar(iterations, values, width=bar_width,
                           color=p["color"], alpha=0.8)

            elif ptype == "overlay_bar":
                if all(is_empty(get_series(s["key"])) for s in p["series"]):
                    ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                            transform=ax.transAxes, color="gray", fontsize=10)
                else:
                    for s_cfg in p["series"]:
                        values = get_series(s_cfg["key"])
                        ax.bar(iterations, values, width=bar_width,
                               label=s_cfg.get("label"),
                               color=s_cfg.get("color"),
                               alpha=s_cfg.get("alpha", 0.8),
                               zorder=s_cfg.get("zorder", 1))
                    ax.legend()

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

            ax.set_ylabel(p.get("ylabel", ""))
            ax.set_title(p["title"])
            ax.grid(True, alpha=0.3)
            # Iteration numbering on the bottom of every panel, every RETRAIN_EVERY iterations
            ax.set_xlabel("Iteration")
            ax.xaxis.set_major_locator(mticker.MultipleLocator(RETRAIN_EVERY))

        for ax, p in zip(axes_flat, PANELS):
            render_panel(ax, p)

        # Hide any unused axes in the final row.
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

        # Save plot image to file
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
            fig.savefig(os.path.join(stats_dir, "run_stats.png"), dpi=200)

            # Also save each panel as its own standalone PNG so individual
            # metrics can be inspected without the surrounding grid.
            panels_dir = os.path.join(stats_dir, "run_stats_panels")
            os.makedirs(panels_dir, exist_ok=True)
            for p in PANELS:
                p_fig, p_ax = plt.subplots(figsize=(7, 4))
                render_panel(p_ax, p)
                slug = re.sub(r"[^a-z0-9]+", "_", p["title"].lower()).strip("_")
                p_fig.tight_layout()
                p_fig.savefig(os.path.join(panels_dir, f"{slug}.png"), dpi=200)
                plt.close(p_fig)
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

    # -- fine-tuning loss plots -----------------------------------------------

    def _plot_finetuning_curve(self, stats_key, ylabel, filename, stats_dir=None):
        """Plot a per-epoch validation curve across every fine-tuning cycle as
        one continuous line.

        ``stats_key`` selects which per-epoch curve to read from each stats
        entry (e.g. ``"finetune_val_loss"`` or ``"finetune_val_kld"``).

        Each fine-tuning contributes only the epochs up to and including its
        saved (best) epoch — the trailing early-stopping patience epochs that
        did not improve the model are excluded, since the checkpointed model is
        the one from the best epoch.

        Segments are laid back-to-back on a shared "cumulative fine-tuning
        epoch" x-axis, coloured per cycle, separated by dashed vertical lines,
        and labelled with the QD iteration at which each fine-tuning happened.
        The saved (best) epoch ending each segment is marked with a ringed dot.
        """
        segments = [
            (s["iteration"], s[stats_key])
            for s in self.stats
            if s.get(stats_key)
        ]
        if not segments:
            log.info("No fine-tuning curves to plot", stats_key=stats_key)
            return

        fig, ax = plt.subplots(figsize=(12, 5))
        cmap = plt.get_cmap("tab10")
        x_offset = 0

        for seg_idx, (iteration, curve) in enumerate(segments):
            xs = np.arange(x_offset, x_offset + len(curve))
            color = cmap(seg_idx % 10)

            ax.plot(xs, curve, color=color, linewidth=1.5,
                    marker="o", markersize=3)
            # Mark the saved (best) epoch — the last point of the segment.
            ax.scatter(xs[-1], curve[-1], color=color, s=45, zorder=3,
                       edgecolor="black", linewidths=0.7)

            # Dashed separator between consecutive fine-tunings.
            if seg_idx > 0:
                ax.axvline(x_offset - 0.5, color="gray", linestyle="--",
                           linewidth=0.8, alpha=0.6)

            # Iteration label pinned to the top of the segment (x in data
            # coords, y in axes fraction).
            ax.text(x_offset, 1.01, f"it {iteration}",
                    transform=ax.get_xaxis_transform(),
                    ha="left", va="bottom", fontsize=8, color=color, rotation=0)

            x_offset += len(curve)

        ax.set_xlabel("Cumulative fine-tuning epoch (saved epochs only)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
            fig.savefig(os.path.join(stats_dir, filename), dpi=200)
        plt.close(fig)

        log.info(
            "Fine-tuning curve plot rendered",
            stats_key=stats_key,
            cycles=len(segments),
            total_saved_epochs=sum(len(c) for _, c in segments),
        )

    def plot_finetuning_val_loss(self, stats_dir=None):
        """Continuous per-epoch validation reconstruction loss across all
        fine-tuning cycles (saved epochs only). See ``_plot_finetuning_curve``."""
        self._plot_finetuning_curve(
            stats_key="finetune_val_loss",
            ylabel="Validation Recon Loss",
            filename="finetuning_val_loss.png",
            stats_dir=stats_dir,
        )

    def plot_finetuning_val_kld(self, stats_dir=None):
        """Continuous per-epoch validation KLD loss across all fine-tuning
        cycles (saved epochs only). See ``_plot_finetuning_curve``."""
        self._plot_finetuning_curve(
            stats_key="finetune_val_kld",
            ylabel="Validation KLD Loss",
            filename="finetuning_val_kld.png",
            stats_dir=stats_dir,
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
