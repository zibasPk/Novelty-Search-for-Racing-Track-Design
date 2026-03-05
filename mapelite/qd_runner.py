# qd_runner.py
# Shared infrastructure for Quality-Diversity search loops.
# Both novelty_search.ipynb and CVT_mapelite.ipynb delegate to these functions.

import os
import glob
import pickle
import datetime
import json

import numpy as np
import matplotlib.pyplot as plt
from ribs.archives import AddStatus
from dask.distributed import Client, LocalCluster

from mapelite.evaluator import EvaluatorMetrics
from mapelite.config import (
    BATCH_SIZE,
    BUFFER_DIR,
    CHECKPOINT_EVERY,
    INVALID_SCORE,
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
            print(f"[Buffer] Resumed {len(self.entries)} entries from {self.buffer_path}")
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
        print(f"[Buffer] Saved {len(self.entries)} entries to {self.buffer_path}")

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


# ── Dask ────────────────────────────────────────────────────────────────────

def setup_dask(batch_size=BATCH_SIZE):
    """Create a Dask LocalCluster and scatter the evaluator to all workers."""
    print("Setting up Dask LocalCluster...")
    cluster = LocalCluster(processes=True, n_workers=batch_size, threads_per_worker=1)
    client = Client(cluster)
    print(f"Dask Dashboard link: {client.dashboard_link}")

    evaluator = EvaluatorMetrics.load_pretrained("mapelite/embeddings/models/model_metrics_VAE/model_metrics_VAE_latent32.pth")
    evaluator_future = client.scatter(evaluator, broadcast=True)
    print(f"Evaluator scattered to {batch_size} Dask workers")

    return client, cluster, evaluator_future


# ── Checkpoint / Resume ─────────────────────────────────────────────────────

def resume_from_checkpoint(checkpoint_dir, stats_dir, stats_filename):
    """
    Try to restore scheduler & stats from the latest checkpoint.

    Returns
    -------
    dict with keys:
        scheduler, archive, start_iter, global_best_score, global_best_id, stats
        - scheduler / archive are *None* when no checkpoint was found (caller
          must build them).
    """
    checkpoints = sorted(glob.glob(f"{checkpoint_dir}checkpoint_*.pkl"))
    start_iter = 0
    global_best_score = INVALID_SCORE
    global_best_id = None
    scheduler = None
    archive = None
    stats = []

    if checkpoints:
        latest_ckpt = checkpoints[-1]
        with open(latest_ckpt, "rb") as f:
            state = pickle.load(f)
        scheduler         = state["scheduler"]
        archive           = scheduler.archive
        start_iter        = state["iteration"]
        global_best_score = state["global_best_score"]
        global_best_id    = state["global_best_id"]
        print(f"[Resume] Loaded {latest_ckpt}, resuming from iteration {start_iter+1}")
    else:
        print("[Resume] No checkpoint found — starting fresh.")

    # Resume stats
    stats_file = os.path.join(stats_dir, stats_filename)
    if os.path.exists(stats_file):
        with open(stats_file, "rb") as f:
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


# ── Main QD loop ────────────────────────────────────────────────────────────

def _eval_on_worker(evaluator, sol):
    """Thin wrapper so Dask receives the evaluator as an already-scattered future."""
    return evaluator.evaluate(sol)


def run_qd_loop(
    scheduler,
    archive,
    client,
    evaluator_future,
    total_iters,
    start_iter,
    checkpoint_dir,
    stats_dir,
    stats_filename,
    stats,
    global_best_score,
    global_best_id,
    buffer_path=None,
):
    """
    Generic ask → evaluate → tell loop used by every QD variant.

    Returns
    -------
    global_best_score, global_best_id, stats  (updated in-place & returned)
    """
    # ── Evaluation buffer setup ──
    if buffer_path is None:
        buffer_path = os.path.join(BUFFER_DIR, "buffer.json")
    evaluation_buffer = EvaluationBuffer(buffer_path)

    for i in range(start_iter, total_iters):

        # Ask the scheduler for a batch of solutions to evaluate
        sols = scheduler.ask()
        sol_dicts = [array_to_solution(sol) for sol in sols]

        # Submit evaluation tasks to Dask and wait for results
        futs = [client.submit(_eval_on_worker, evaluator_future, sol)
                for sol in sol_dicts]
        gathered = [f.result() for f in futs]

        obj_list, clean_solutions = [], []
        for (sol_id, ok, msg, score, desc), sol_dict in zip(gathered, sol_dicts):

            if not ok:
                print(f"Warning: clamping bad score for ID={sol_id} ({msg})")
                score = INVALID_SCORE
            else:
                print(f"Solution ID={sol_id} evaluated with score={score:.2f}")
                if score > global_best_score:
                    global_best_score = score
                    global_best_id = sol_id

            # Record every evaluation in the buffer
            evaluation_buffer.record(sol_id, sol_dict, desc, score, ok)

            clean_solutions.append((score, desc))
            obj_list.append(score)

        obj_batch, meas_batch = zip(*clean_solutions)

        # ── Track new / substituted elites via temporary monkey-patch ──
        new_elites_count = 0
        sub_elites_count = 0
        original_add = archive.add

        def tracked_add(*args, **kwargs):
            nonlocal new_elites_count, sub_elites_count
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
                new_elites_count += int(np.sum(arr == AddStatus.NEW))
                sub_elites_count += int(np.sum(arr == AddStatus.IMPROVE_EXISTING))

            return res

        archive.add = tracked_add
        try:
            scheduler.tell(list(obj_batch), list(meas_batch))
        finally:
            if "add" in archive.__dict__:
                del archive.__dict__["add"]

        # ── Logging ──
        batch_best = max(obj_list) if obj_list else INVALID_SCORE
        print(f"Iteration {i+1} ended. Best in batch = {batch_best:.2f}")
        print(f"Global best so far: {global_best_score:.2f} (ID={global_best_id})")
        print(f"Archive Updates: {new_elites_count} new elites inserted, "
              f"{sub_elites_count} elites substituted.")

        data_archive = archive.data()
        arch_obj = data_archive["objective"]
        valid = arch_obj != INVALID_SCORE
        mean_val = np.mean(arch_obj[valid]) if np.any(valid) else 0.0
        best_val = np.max(arch_obj[valid]) if np.any(valid) else 0.0
        elites = archive.stats.num_elites
        print(f"Archive size={elites}, mean={mean_val:.2f}, best={best_val:.2f}")

        stats.append({
            "iteration": i,
            "Archive size": elites,
            "iteration_best": batch_best,
            "global_best_score": global_best_score,
            "new_elites": new_elites_count,
            "substituted_elites": sub_elites_count,
        })

        # ── Checkpoint ──
        if (i + 1) % CHECKPOINT_EVERY == 0:
            ckpt_name = f"{checkpoint_dir}checkpoint_{i+1:04d}.pkl"
            with open(ckpt_name, "wb") as f:
                pickle.dump({
                    "scheduler": scheduler,
                    "iteration": i + 1,
                    "global_best_score": global_best_score,
                    "global_best_id": global_best_id,
                }, f)
            print(f"[Checkpoint] Saved {ckpt_name}")

            stats_path = os.path.join(stats_dir, stats_filename)
            with open(stats_path, "wb") as f:
                pickle.dump(stats, f)

            # Persist the evaluation buffer alongside each checkpoint
            evaluation_buffer.save()

    # Final save to ensure nothing is lost
    evaluation_buffer.save()

    return global_best_score, global_best_id, stats


# ── Visualization ───────────────────────────────────────────────────────────

def plot_stats(stats, title="QD Run Statistics"):
    """5-row run-statistics plot (archive growth, fitness, new elites, substituted elites, cumulative)."""

    iterations    = [s["iteration"] + 1 for s in stats]
    archive_sizes = [s["Archive size"] for s in stats]
    iter_best     = [s["iteration_best"] for s in stats]
    global_best   = [s["global_best_score"] for s in stats]
    new_elites    = [s["new_elites"] for s in stats]
    sub_elites    = [s["substituted_elites"] for s in stats]

    # Filter out INVALID_SCORE so the y-axis isn't crushed
    iter_best_clean = [v if v != INVALID_SCORE else np.nan for v in iter_best]

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f"{title} — Run Statistics", fontsize=16, fontweight="bold")

    # 1. Archive size
    ax = axes[0]
    ax.plot(iterations, archive_sizes, color="tab:blue", linewidth=1.5)
    ax.set_ylabel("Archive Size")
    ax.set_title("Archive Growth")
    ax.grid(True, alpha=0.3)

    # 2. Best fitness
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

    # 3. New elites per iteration
    ax = axes[2]
    ax.bar(iterations, new_elites, width=bar_width,
           color="tab:red", alpha=0.8)
    ax.set_ylabel("Count")
    ax.set_title("New Elites per Iteration")
    ax.grid(True, alpha=0.3)

    # 4. Substituted elites per iteration
    ax = axes[3]
    ax.bar(iterations, sub_elites, width=bar_width,
           color="tab:blue", alpha=0.8)
    ax.set_ylabel("Count")
    ax.set_title("Substituted Elites per Iteration")
    ax.grid(True, alpha=0.3)

    # 5. Cumulative elites
    ax = axes[4]
    cum_new = np.cumsum(new_elites)
    cum_sub = np.cumsum(sub_elites)
    ax.plot(iterations, cum_new, label="Cumulative New",
            color="tab:green", linewidth=1.5)
    ax.plot(iterations, cum_sub, label="Cumulative Substituted",
            color="tab:purple", linewidth=1.5)
    ax.plot(iterations, cum_new + cum_sub, label="Cumulative Total",
            color="tab:blue", linewidth=2, linestyle="--")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative Count")
    ax.set_title("Cumulative Elite Insertions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # Summary table
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


# ── Elite export ────────────────────────────────────────────────────────────

def export_elites(
    archive,
    stats,
    output_path,
    algorithm_label,
    seed,
    global_best_score,
    global_best_id,
):
    """Save all valid elites to a JSON file for reconstruction & visualization."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data_archive = archive.data()
    solutions  = np.array(data_archive["solution"])
    objectives = np.array(data_archive["objective"])
    measures   = np.array(data_archive["measures"])

    elites_list = []
    skipped_invalid = 0
    for idx in range(len(objectives)):
        fit = float(objectives[idx])
        if not np.isfinite(fit) or fit == INVALID_SCORE:
            skipped_invalid += 1
            continue

        sol_arr  = solutions[idx]
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
        print(f"Skipped {skipped_invalid} elites with invalid fitness (INVALID_SCORE or NaN)")

    elites_list.sort(key=lambda e: e["fitness"], reverse=True)

    output = {
        "metadata": {
            "algorithm":    algorithm_label,
            "totalElites":  len(elites_list),
            "embeddingDim": int(measures.shape[1]),
            "solutionDim":  int(solutions.shape[1]),
            "seed":         seed,
            "iterations":   len(stats),
            "globalBest":   float(global_best_score),
            "globalBestId": global_best_id,
            "timestamp":    datetime.datetime.now().isoformat(),
        },
        "elites": elites_list,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(elites_list)} elites to {output_path}")
    print(f"  Best fitness:  {elites_list[0]['fitness']:.4f} (ID={elites_list[0]['id']})")
    print(f"  Worst fitness: {elites_list[-1]['fitness']:.4f}")
    print(f"  File size:     {os.path.getsize(output_path) / 1024:.1f} KB")
