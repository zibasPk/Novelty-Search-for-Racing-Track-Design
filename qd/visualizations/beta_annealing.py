"""Illustrate the cyclical KLD (beta) annealing schedule used when training the VAE.

Run with ``python -m qd.visualizations.beta_annealing``. Produces a PNG in
``qd/visualizations/plots/beta_annealing/``:

- ``beta_annealing.png`` — beta vs. epoch for the base training and fine-tuning
  configs, with the linear warm-up ramp shaded.

The schedule mirrors ``VAETrainer._compute_beta``: within each cycle beta ramps
linearly from 0 to ``max_beta`` over the first ``ratio`` fraction of the cycle,
then stays flat at ``max_beta`` until the cycle restarts.
"""

import os

import matplotlib
import matplotlib.pyplot as plt

from qd.vae.config import FINETUNING_CONFIG, TRAINING_CONFIG

matplotlib.use("Agg")

# ── Style ──────────────────────────────────────────────────────────────────

BASE_COLOR = "#e0792e"
FINETUNE_COLOR = "#2f6f6f"
RAMP_FACE = "#f7cf8b"

OUT_DIR = os.path.join("qd", "visualizations", "plots", "beta_annealing")


def compute_beta(epoch: int, epochs: int, n_cycles: int, max_beta: float, ratio: float) -> float:
    """Beta for a given epoch — identical to ``VAETrainer._compute_beta``."""
    cycle_len = max(1, epochs // n_cycles)
    cycle_idx = epoch % cycle_len
    if cycle_idx < cycle_len * ratio:
        return max_beta * (cycle_idx / (cycle_len * ratio))
    return max_beta


def shade_ramps(ax, epochs: int, n_cycles: int, ratio: float) -> None:
    """Shade the linear warm-up portion of every cycle."""
    cycle_len = max(1, epochs // n_cycles)
    labelled = False
    for c in range(n_cycles):
        start = c * cycle_len
        end = start + cycle_len * ratio
        ax.axvspan(
            start,
            end,
            color=RAMP_FACE,
            alpha=0.5,
            lw=0,
            label="warm-up ramp" if not labelled else None,
        )
        labelled = True


def plot_schedule(ax, cfg: dict, title: str, color: str) -> None:
    epochs = cfg["epochs"]
    kld = cfg["kld"]
    n_cycles = kld["n_cycles"]
    max_beta = kld["max_beta"]
    ratio = kld["ratio"]

    epoch_range = list(range(epochs))
    betas = [compute_beta(e, epochs, n_cycles, max_beta, ratio) for e in epoch_range]

    shade_ramps(ax, epochs, n_cycles, ratio)
    ax.plot(epoch_range, betas, color=color, lw=2.2)
    ax.axhline(max_beta, color="#888888", ls="--", lw=1, label=f"max_beta = {max_beta:g}")

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\beta$ (KLD weight)")
    ax.set_xlim(0, epochs - 1)
    ax.set_ylim(0, max_beta * 1.25)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    annotation = (
        f"n_cycles = {n_cycles}\n"
        f"ratio = {ratio:g}  (warm-up = {int(ratio * 100)}% of cycle)"
    )
    ax.text(
        0.02,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9),
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_schedule(axes[0], TRAINING_CONFIG, "Base training", BASE_COLOR)
    plot_schedule(axes[1], FINETUNING_CONFIG, "Fine-tuning", FINETUNE_COLOR)

    fig.suptitle("Cyclical KLD ($\\beta$) annealing schedule", fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "beta_annealing.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
