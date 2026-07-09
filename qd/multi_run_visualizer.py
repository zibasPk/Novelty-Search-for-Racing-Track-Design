# multi_run_visualizer.py
# Aggregate run-statistics visualizations across several QD runs.
#
# This is the multi-run counterpart to ``ArchiveVisualizer.plot_stats``: it reads
# the per-iteration ``stats`` list from each run and overlays them on shared axes,
# adding an across-runs mean and a ±1 std band per metric.
#
# Only the ``stats`` list is used (the same data plot_stats reads). The other
# ArchiveVisualizer plots (UMAP heatmap, archive grid, elite track images) depend
# on the live pyribs archive and the reconstruct API, so they are per-run
# artifacts with no meaningful cross-run aggregation and are intentionally not
# reproduced here.
#
# Usage (from repo root):
#     python -m qd.multi_run_visualizer
#     python -m qd.multi_run_visualizer --runs-dir data/multi_run --out data/multi_run/_plots

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import pickle
import re
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Config constants, with standalone fallbacks so the script also runs outside the
# package (the loader stubs out QD/ribs classes, so no heavy deps are required).
try:
    from qd.config import RETRAIN_EVERY, INVALID_SCORE
except Exception:  # pragma: no cover - fallback when run detached from the package
    RETRAIN_EVERY, INVALID_SCORE = 100, -1e9


# ── Tolerant checkpoint loading ─────────────────────────────────────────────
#
# Checkpoints pickle the whole scheduler (ProximityArchive, emitters, …). We only
# want the plain ``stats`` list, so we stub out the heavy custom classes during
# unpickling. Everything importable (numpy, builtins) still resolves normally.

_STUB_MODULE_ROOTS = {"ribs", "qd", "dask", "distributed", "torch", "sklearn"}


class _Stub:
    """Absorbs any constructor args / pickle state without importing the real class."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *args, **kwargs):
        return _Stub()


class _TolerantUnpickler(pickle.Unpickler):
    """Unpickler that replaces unavailable/heavy classes with a harmless stub.

    The top-level checkpoint object is a plain ``dict`` and ``stats`` is a list of
    dicts of plain scalars, so stubbing the scheduler object graph never touches
    the data we care about.
    """

    def find_class(self, module, name):
        if module.split(".")[0] in _STUB_MODULE_ROOTS:
            return _Stub
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub


def _extract_stats(obj):
    """Pull the ``stats`` list out of a loaded checkpoint/stats object."""
    if isinstance(obj, dict) and "stats" in obj:
        obj = obj["stats"]
    if isinstance(obj, list):
        return obj
    raise ValueError("Loaded object does not contain a 'stats' list")


def _load_stats_from_file(path):
    """Load a stats list from a ``.pkl`` (checkpoint or bare) or ``.json`` file."""
    if path.lower().endswith(".json"):
        with open(path, "r") as f:
            return _extract_stats(json.load(f))
    with open(path, "rb") as f:
        return _extract_stats(_TolerantUnpickler(f).load())


def _latest_checkpoint(pkls):
    """Return the checkpoint path with the highest trailing iteration number."""
    def it_num(p):
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    return max(pkls, key=it_num)


def _resolve_run_source(path):
    """Given a run directory or file, return the concrete file to load, or None."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        return None
    ckpts = glob.glob(os.path.join(path, "checkpoint_*.pkl"))
    if ckpts:
        return _latest_checkpoint(ckpts)
    for name in ("stats.pkl", "stats.json"):
        cand = os.path.join(path, name)
        if os.path.isfile(cand):
            return cand
    loose = sorted(glob.glob(os.path.join(path, "*.pkl")))
    if loose:
        return _latest_checkpoint(loose)
    return None


def discover_runs(runs_dir):
    """Find runs under *runs_dir*.

    Each sub-directory is one run (labelled by folder name); loose top-level
    ``*.pkl`` / ``stats.json`` files are each treated as one run too. Returns a
    list of ``(label, stats_list)`` sorted by label.
    """
    runs = []
    entries = sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []

    for entry in entries:
        full = os.path.join(runs_dir, entry)
        if os.path.isdir(full):
            src = _resolve_run_source(full)
            label = entry
        elif entry.lower().endswith((".pkl", ".json")):
            src = full
            label = os.path.splitext(entry)[0]
        else:
            continue

        if src is None:
            print(f"  [skip] {entry}: no checkpoint/stats file found")
            continue
        try:
            stats = _load_stats_from_file(src)
        except Exception as exc:
            print(f"  [skip] {entry}: failed to load ({exc})")
            continue
        if not stats:
            print(f"  [skip] {entry}: empty stats")
            continue
        runs.append((label, stats))
        print(f"  [ok]   {label}: {len(stats)} iterations  <- {os.path.relpath(src, runs_dir)}")

    return sorted(runs, key=lambda r: r[0])


# ── Aggregation helpers ─────────────────────────────────────────────────────

def _run_series(stats, key, clean_invalid):
    """Map iteration -> float value for *key* in one run (invalid/None -> NaN)."""
    out = {}
    for s in stats:
        it = s.get("iteration")
        if it is None:
            continue
        v = s.get(key, np.nan)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = np.nan
        if clean_invalid and v == INVALID_SCORE:
            v = np.nan
        out[int(it)] = v
    return out


def _moving_average(y, window):
    """NaN-aware centred moving average; returns *y* unchanged if window <= 1."""
    y = np.asarray(y, dtype=float)
    if window <= 1:
        return y
    half = window // 2
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        seg = y[max(0, i - half): i + half + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def _build_matrix(runs, key, clean_invalid):
    """Build a (n_runs, n_iters) matrix aligned on the union iteration axis."""
    per_run = [_run_series(stats, key, clean_invalid) for _, stats in runs]
    iters = sorted({it for d in per_run for it in d})
    if not iters:
        return np.array([]), np.empty((len(runs), 0))
    idx = {it: j for j, it in enumerate(iters)}
    mat = np.full((len(runs), len(iters)), np.nan)
    for r, d in enumerate(per_run):
        for it, v in d.items():
            mat[r, idx[it]] = v
    return np.asarray(iters), mat


# ── Panel definitions ───────────────────────────────────────────────────────
# key          – stats key to read
# clean_invalid – map INVALID_SCORE -> NaN before plotting (fitness-like metrics)
# smooth        – apply the moving-average window to per-run + mean lines
# sparse        – metric recorded only on some iterations (e.g. retrain); plot as
#                 markers on available points, no smoothing, no std band

PANELS = [
    {"title": "Archive Growth", "ylabel": "Archive Size", "key": "Archive size"},
    {"title": "Global Best Fitness", "ylabel": "Fitness", "key": "global_best_score",
     "clean_invalid": True},
    {"title": "Mean Archive Fitness", "ylabel": "Mean Fitness", "key": "mean_fitness"},
    {"title": "Iteration Best", "ylabel": "Fitness", "key": "iteration_best",
     "clean_invalid": True, "smooth": True},
    {"title": "New Elites per Iteration", "ylabel": "Count", "key": "new_elites",
     "smooth": True},
    {"title": "Substituted Elites per Iteration", "ylabel": "Count",
     "key": "substituted_elites", "smooth": True},
    {"title": "Archive Acceptance Rate", "ylabel": "Acceptance Rate",
     "key": "acceptance_rate", "smooth": True},
    {"title": "High-Quality Coverage", "ylabel": "Count", "key": "high_quality_coverage"},
    {"title": "Fitness–Novelty Correlation", "ylabel": "Pearson r",
     "key": "fitness_novelty_corr"},
    {"title": "Reconstruction Loss after Retraining", "ylabel": "Val Recon Loss",
     "key": "recon_loss", "sparse": True},
]


def _nan_mean_std(mat):
    """nanmean/nanstd/finite-count along axis 0, silencing all-NaN-slice warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
    count = np.sum(np.isfinite(mat), axis=0)
    return mean, std, count


def _render_panel(ax, panel, runs, run_colors, smooth_window):
    """Draw one metric panel: per-run lines + across-runs mean and ±1 std band."""
    key = panel["key"]
    iters, mat = _build_matrix(runs, key, panel.get("clean_invalid", False))

    if mat.size == 0 or not np.isfinite(mat).any():
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=10)
        _finish_axis(ax, panel)
        return

    if panel.get("sparse"):
        # Markers on the iterations each run actually recorded a value.
        for r, (label, _) in enumerate(runs):
            row = mat[r]
            m = np.isfinite(row)
            if m.any():
                ax.plot(iters[m], row[m], color=run_colors[r], marker="o",
                        markersize=3, linewidth=1.0, alpha=0.7)
        mean, _, _ = _nan_mean_std(mat)
        mmask = np.isfinite(mean)
        if mmask.any():
            ax.plot(iters[mmask], mean[mmask], color="black", linewidth=2.0,
                    marker="o", markersize=4, zorder=5, label="mean")
        _finish_axis(ax, panel)
        return

    plot_mat = mat
    if panel.get("smooth") and smooth_window > 1:
        plot_mat = np.vstack([_moving_average(row, smooth_window) for row in mat])

    # Per-run lines.
    for r, (label, _) in enumerate(runs):
        ax.plot(iters, plot_mat[r], color=run_colors[r], alpha=0.40, linewidth=1.0)

    # Across-runs mean + ±1 std band (band only where >= 2 runs contribute).
    mean, std, count = _nan_mean_std(plot_mat)
    band = count >= 2
    if band.any():
        ax.fill_between(iters, mean - std, mean + std, where=band,
                        color="black", alpha=0.12, linewidth=0, zorder=3)
    mmask = np.isfinite(mean)
    ax.plot(iters[mmask], mean[mmask], color="black", linewidth=2.0,
            zorder=4, label="mean")
    _finish_axis(ax, panel)


def _finish_axis(ax, panel):
    ax.set_ylabel(panel.get("ylabel", ""))
    ax.set_title(panel["title"])
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(RETRAIN_EVERY))


def plot_multi_run_stats(runs, out_dir, smooth_window=11, title="Multi-Run QD Statistics"):
    """Render the combined grid, per-panel PNGs, and a final-metrics CSV."""
    os.makedirs(out_dir, exist_ok=True)

    cmap = plt.get_cmap("tab10" if len(runs) <= 10 else "tab20")
    run_colors = [cmap(i % cmap.N) for i in range(len(runs))]

    n_cols = 2
    n_rows = (len(PANELS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3), squeeze=False)
    axes_flat = axes.flatten()
    fig.suptitle(f"{title} — {len(runs)} runs", fontsize=16, fontweight="bold")

    for ax, panel in zip(axes_flat, PANELS):
        _render_panel(ax, panel, runs, run_colors, smooth_window)
    for ax in axes_flat[len(PANELS):]:
        ax.set_visible(False)

    # One shared legend mapping colours to run labels (+ the mean line).
    handles = [plt.Line2D([], [], color=run_colors[r], label=label)
               for r, (label, _) in enumerate(runs)]
    handles.append(plt.Line2D([], [], color="black", linewidth=2.0, label="mean"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    combined_path = os.path.join(out_dir, "multi_run_stats.png")
    fig.savefig(combined_path, dpi=200)
    plt.close(fig)

    # Standalone per-panel PNGs.
    panels_dir = os.path.join(out_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)
    for panel in PANELS:
        p_fig, p_ax = plt.subplots(figsize=(7, 4))
        _render_panel(p_ax, panel, runs, run_colors, smooth_window)
        p_ax.legend(handles=handles, fontsize=7, ncol=2, frameon=False)
        slug = re.sub(r"[^a-z0-9]+", "_", panel["title"].lower()).strip("_")
        p_fig.tight_layout()
        p_fig.savefig(os.path.join(panels_dir, f"{slug}.png"), dpi=200)
        plt.close(p_fig)

    csv_path = _write_final_metrics_csv(runs, out_dir)

    print(f"\nSaved:\n  {combined_path}\n  {panels_dir}/<metric>.png\n  {csv_path}")
    return combined_path


def _write_final_metrics_csv(runs, out_dir):
    """Write the final-iteration value of each panel metric, per run + mean/std."""
    keys = [p["key"] for p in PANELS]
    rows = []
    for label, stats in runs:
        last = stats[-1] if stats else {}
        row = {"run": label, "iterations": len(stats)}
        for k in keys:
            v = last.get(k, "")
            # recon_loss etc. may be absent on the final iteration; fall back to
            # the last iteration that recorded the metric.
            if v in ("", None):
                for s in reversed(stats):
                    if s.get(k) not in (None, ""):
                        v = s[k]
                        break
            row[k] = v if not isinstance(v, list) else (v[-1] if v else "")
        rows.append(row)

    csv_path = os.path.join(out_dir, "multi_run_final_metrics.csv")
    fieldnames = ["run", "iterations"] + keys
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return csv_path


# ── Fine-tuning curve aggregation ───────────────────────────────────────────
#
# The per-epoch fine-tuning curves (``finetune_val_loss`` / ``finetune_val_kld``)
# can't be laid back-to-back across runs the way ArchiveVisualizer does within a
# single run: early stopping makes each cycle's epoch count differ across runs.
# Instead we align on the *fine-tuning cycle* (keyed by the QD iteration it ran
# at, which is shared across runs) and aggregate two ways:
#   1. normalized within-cycle shape (epoch axis rescaled to progress 0..1), and
#   2. per-cycle scalars (start / best loss / #epochs) vs iteration.

def _collect_finetune_cycles(runs, stats_key):
    """Map cycle iteration -> list of (run_idx, curve) for every run that ran it."""
    cycles = {}
    for r, (_, stats) in enumerate(runs):
        for s in stats:
            curve = s.get(stats_key)
            if curve:  # non-empty list only (recorded on retraining iterations)
                it = int(s["iteration"])
                cycles.setdefault(it, []).append((r, [float(x) for x in curve]))
    return cycles


def plot_finetuning_normalized(runs, out_dir, stats_key, ylabel, filename, n_grid=25):
    """Overlay the across-runs mean convergence curve of every fine-tuning cycle.

    Each run's variable-length epoch curve is resampled onto a common
    normalized-progress grid (0 = first epoch, 1 = saved/best epoch), then
    averaged across runs per cycle. Cycles are coloured by QD iteration so the
    drift of the convergence shape over the run is visible; the shaded band is
    ±1 std across runs.
    """
    cycles = _collect_finetune_cycles(runs, stats_key)
    if not cycles:
        return None

    iters_sorted = sorted(cycles)
    grid = np.linspace(0.0, 1.0, n_grid)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(iters_sorted), max(iters_sorted)) \
        if len(iters_sorted) > 1 else plt.Normalize(iters_sorted[0] - 1, iters_sorted[0] + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for it in iters_sorted:
        resampled = []
        for _, curve in cycles[it]:
            if len(curve) >= 2:
                p = np.linspace(0.0, 1.0, len(curve))
                resampled.append(np.interp(grid, p, curve))
            elif len(curve) == 1:  # single saved epoch — flat line at that value
                resampled.append(np.full(n_grid, curve[0]))
        if not resampled:
            continue
        mat = np.vstack(resampled)
        mean = mat.mean(axis=0)
        color = cmap(norm(it))
        ax.plot(grid, mean, color=color, linewidth=1.8, zorder=3)
        if mat.shape[0] >= 2:
            std = mat.std(axis=0)
            ax.fill_between(grid, mean - std, mean + std, color=color,
                            alpha=0.15, linewidth=0, zorder=2)

    ax.set_xlabel("Normalized fine-tuning progress (0 = first epoch, 1 = saved/best epoch)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Mean fine-tuning convergence across {len(runs)} runs")
    ax.grid(True, alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label("QD iteration (fine-tuning cycle)")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_finetuning_summary(runs, out_dir, filename="finetuning_summary.png"):
    """Per-cycle scalar summary vs QD iteration, aggregated across runs.

    Left panel: mean ±1 std of the first-epoch and best-epoch validation recon
    loss per cycle (the gap between them is how much each fine-tune helped).
    Right panel: mean ±1 std of the number of epochs to the saved/best model.
    """
    cycles = _collect_finetune_cycles(runs, "finetune_val_loss")
    if not cycles:
        return None

    iters = np.array(sorted(cycles))

    def _agg(reduce_fn):
        means, stds = [], []
        for it in iters:
            vals = [reduce_fn(curve) for _, curve in cycles[it]]
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)
        return np.array(means), np.array(stds)

    start_m, start_s = _agg(lambda c: c[0])
    end_m, end_s = _agg(lambda c: c[-1])
    ep_m, ep_s = _agg(len)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(iters, start_m, color="tab:red", marker="o", label="first epoch")
    ax1.fill_between(iters, start_m - start_s, start_m + start_s,
                     color="tab:red", alpha=0.15, linewidth=0)
    ax1.plot(iters, end_m, color="tab:green", marker="o", label="best (saved) epoch")
    ax1.fill_between(iters, end_m - end_s, end_m + end_s,
                     color="tab:green", alpha=0.15, linewidth=0)
    ax1.set_title("Validation recon loss per fine-tuning cycle")
    ax1.set_ylabel("Val Recon Loss")
    ax1.legend(fontsize=8)

    ax2.plot(iters, ep_m, color="tab:blue", marker="o")
    ax2.fill_between(iters, ep_m - ep_s, ep_m + ep_s,
                     color="tab:blue", alpha=0.15, linewidth=0)
    ax2.set_title("Epochs to saved (best) model per cycle")
    ax2.set_ylabel("Epochs")

    for ax in (ax1, ax2):
        ax.set_xlabel("QD iteration (fine-tuning cycle)")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(RETRAIN_EVERY))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_finetuning(runs, out_dir):
    """Render all fine-tuning aggregate plots; returns the list of saved paths."""
    saved = [
        plot_finetuning_normalized(
            runs, out_dir, "finetune_val_loss",
            "Validation Recon Loss", "finetuning_val_loss_normalized.png"),
        plot_finetuning_normalized(
            runs, out_dir, "finetune_val_kld",
            "Validation KLD Loss", "finetuning_val_kld_normalized.png"),
        plot_finetuning_summary(runs, out_dir),
    ]
    return [p for p in saved if p]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="data/multi_run",
                        help="Folder containing one sub-folder (or loose pkl/json) per run.")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: <runs-dir>/_plots).")
    parser.add_argument("--smooth", type=int, default=11,
                        help="Moving-average window for noisy per-iteration metrics.")
    parser.add_argument("--title", default="Multi-Run QD Statistics")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.runs_dir, "_plots")

    print(f"Scanning {args.runs_dir} for runs...")
    runs = discover_runs(args.runs_dir)
    if not runs:
        print("No runs found. See data/multi_run/README.md for the expected layout.")
        return

    print(f"\nAggregating {len(runs)} runs -> {out_dir}")
    plot_multi_run_stats(runs, out_dir, smooth_window=args.smooth, title=args.title)

    ft_paths = plot_finetuning(runs, out_dir)
    if ft_paths:
        print("  " + "\n  ".join(ft_paths))
    else:
        print("  (no fine-tuning curves found in these runs — skipped)")


if __name__ == "__main__":
    main()
