"""Reconstruction-quality time-series check on the latest archive elites.

This mirrors the "Reconstruction Quality — Random Track Time Series" diagnostic,
but instead of random test-set tracks it reconstructs a handful of *current
archive elites* and compares the *latest finetuned model* produced by the QD run
against the *initial pretrained model* (before any finetuning).

For each sampled elite we recover its raw phenotype sequence from the run's
evaluation buffer (keyed by solution id), preprocess it exactly as the runner
does (``MetricsPreprocessor``), pass it through *both* VAEs, phase-align each
reconstruction to the original (the VAE is shift-invariant, so an arbitrary
circular phase offset is expected), and plot the original alongside both
reconstructions for each of the three features.

Run from the repository root:

    python -m qd.analysis.results_analysis.reconstruction_quality_elites
    python -m qd.analysis.results_analysis.reconstruction_quality_elites --n-tracks 8 --seed 67
    python -m qd.analysis.results_analysis.reconstruction_quality_elites --model data/ns/checkpoints/finetuned_model_1100.pt
    python -m qd.analysis.results_analysis.reconstruction_quality_elites --initial-model qd/embeddings/models/...pth
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast
from matplotlib.lines import Line2D

from torch.utils.data import DataLoader

from qd.vae import MetricsVAE, MetricsDataset, MetricsPreprocessor
from qd.vae.data import collate_fn
from qd.vae.losses import vae_loss as shift_invariant_vae_loss_fn
from qd.config import EMBEDDING_MODEL_PATH

# ── Config ────────────────────────────────────────────────────────────────────
FEATURE_NAMES = ['Speed', 'Steering', 'Track Position']
Y_LIMITS      = [(0, 1), (-0.5, 0.5), (-0.5, 0.5)]

DEFAULT_CHECKPOINT_DIR = "data/ns/checkpoints"
DEFAULT_ELITES_FILE    = "data/ns/elites.json"
DEFAULT_BUFFER_FILE    = "data/ns/buffer.json"
DEFAULT_INITIAL_MODEL  = EMBEDDING_MODEL_PATH  # pretrained model, before any finetuning


def latest_finetuned_model(checkpoint_dir):
    """Return the path of the most recent ``finetuned_model_*.pt`` checkpoint."""
    paths = sorted(glob.glob(os.path.join(checkpoint_dir, "finetuned_model_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No finetuned_model_*.pt found in {checkpoint_dir}")
    return paths[-1]


def build_elite_dataset(elites_file, buffer_file):
    """Build a MetricsDataset over the elites' preprocessed phenotype sequences.

    The elites carry only their genotype + embedding; their raw phenotype
    telemetry lives in the evaluation buffer, keyed by solution id. We join the
    two and preprocess each sequence (dropping any that fail validation,
    mirroring ``QDRunner._finetune_embedding_model``). Returns ``(dataset, seed)``
    where ``seed`` is the run seed recorded in the elites metadata.
    """
    with open(elites_file, "r") as f:
        elites_data = json.load(f)
    with open(buffer_file, "r") as f:
        buffer_data = json.load(f)

    phenotype_by_id = {t["id"]: t.get("phenotype_data") for t in buffer_data["tracks"]}
    elites = elites_data["elites"]
    run_seed = elites_data["metadata"].get("seed")
    print(f"Loaded {len(elites)} elites (totalElites="
          f"{elites_data['metadata'].get('totalElites')})")

    preprocessor = MetricsPreprocessor()

    def _safe_preprocess(raw):
        try:
            return preprocessor(np.array(raw, dtype=np.float32))
        except Exception:
            return None

    processed = []
    missing = 0
    failed = 0
    for elite in elites:
        raw = phenotype_by_id.get(elite["id"])
        if raw is None:
            missing += 1
            continue
        p = _safe_preprocess(raw)
        if p is None:
            failed += 1
            continue
        processed.append(p)

    print(f"Phenotype available: {len(processed)} | missing in buffer: {missing} "
          f"| failed preprocessing: {failed}")
    if not processed:
        raise ValueError("No valid elite phenotype sequences available to reconstruct.")

    return MetricsDataset(processed), run_seed


def align_signals(orig, recon):
    """Circularly shift ``recon`` to best match ``orig`` (the VAE is shift-invariant)."""
    T, C = orig.shape
    O = np.fft.rfft(orig, axis=0)
    R = np.fft.rfft(recon, axis=0)
    cc = np.fft.irfft(O * np.conj(R), n=T, axis=0).sum(axis=-1)
    best_shift = int(np.argmax(cc))
    return np.roll(recon, shift=best_shift, axis=0)


def reconstruct(model, sample, orig, device, beta, dim_weights):
    """Reconstruct ``sample`` through ``model`` and phase-align to ``orig``.

    Returns ``(recon, recon_loss)`` with ``recon`` shaped ``(T, C)``.
    """
    with torch.no_grad(), autocast(device.type):
        recon_x, mu, log_var = model(sample)
        _, recon_loss, _ = shift_invariant_vae_loss_fn(
            recon_x, sample, mu, log_var, beta=beta, dim_weights=dim_weights
        )
    recon = recon_x[0].float().cpu().numpy()
    recon = align_signals(orig, recon)
    return recon, recon_loss.item()


def batch_recon_losses(model, dataset, device, dim_weights, chunk=12, label=""):
    """Per-sample shift-invariant reconstruction loss for every item in ``dataset``.

    Uses the project's ``collate_fn`` (padding + mask) so variable-length elite
    sequences are handled correctly, batches the forward pass so ranking all
    elites takes a handful of passes instead of one per sample, then reuses the
    real ``vae_loss`` recon term per sample (cheap — only the FFTs, not a forward).
    """
    loader = DataLoader(dataset, batch_size=chunk, shuffle=False, collate_fn=collate_fn)
    n_batches = len(loader)
    print(f"Scoring {len(dataset)} elites with {label or 'model'} "
          f"({n_batches} batch(es) of up to {chunk})...")
    losses = []
    with torch.no_grad():
        for b, (padded, mask) in enumerate(loader, start=1):
            padded = padded.to(device)
            mask   = mask.to(device)
            with autocast(device.type):
                recon_x, mu, log_var = model(padded, src_key_padding_mask=mask)

            lengths = (~mask).sum(dim=1)
            for i in range(padded.size(0)):
                L = int(lengths[i].item())
                _, recon_loss, _ = shift_invariant_vae_loss_fn(
                    recon_x[i:i + 1, :L], padded[i:i + 1, :L],
                    mu[i:i + 1], log_var[i:i + 1],
                    beta=0.0, dim_weights=dim_weights,
                )
                losses.append(recon_loss.item())
            print(f"  [{label or 'model'}] batch {b}/{n_batches} "
                  f"({len(losses)}/{len(dataset)} elites scored)", flush=True)

    return np.array(losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="Path to the finetuned VAE checkpoint "
                             "(default: latest finetuned_model_*.pt under --checkpoint-dir)")
    parser.add_argument("--initial-model", default=DEFAULT_INITIAL_MODEL,
                        help="Path to the initial pretrained VAE checkpoint, before any "
                             f"finetuning (default: {DEFAULT_INITIAL_MODEL})")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--elites", default=DEFAULT_ELITES_FILE)
    parser.add_argument("--buffer", default=DEFAULT_BUFFER_FILE)
    parser.add_argument("--n-tracks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for picking elites (default: a fresh random seed each run, "
                             "so the displayed tracks change every time)")
    parser.add_argument("--top-improved", action="store_true",
                        help="Instead of random elites, show the top --n-tracks elites by "
                             "reconstruction improvement (initial recon loss minus finetuned).")
    parser.add_argument("--save-fig", default=None,
                        help="Optional path to save the figure instead of showing it")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model or latest_finetuned_model(args.checkpoint_dir)
    print(f"Using finetuned model: {model_path}")
    model, latent_dim, parameters = MetricsVAE.load_pretrained(model_path, device)

    print(f"Using initial (pretrained) model: {args.initial_model}")
    initial_model, _, _ = MetricsVAE.load_pretrained(args.initial_model, device)

    elite_dataset, _ = build_elite_dataset(args.elites, args.buffer)

    beta        = parameters["kld"]["max_beta"]
    dim_weights = parameters.get("dim_weights", None)
    if dim_weights is not None:
        dim_weights = torch.tensor(dim_weights, dtype=torch.float32, device=device)

    model.eval()
    initial_model.eval()

    n_tracks = min(args.n_tracks, len(elite_dataset))

    if args.top_improved:
        # Rank every elite by how much the finetuned model lowered its
        # reconstruction loss vs. the initial model, then take the top N.
        loss_ft   = batch_recon_losses(model,         elite_dataset, device, dim_weights, label="finetuned")
        loss_init = batch_recon_losses(initial_model, elite_dataset, device, dim_weights, label="initial")
        improvements = loss_init - loss_ft
        track_indices = np.argsort(improvements)[::-1][:n_tracks]
        print(f"Top {n_tracks} elites by recon improvement (initial - finetuned): "
              + ", ".join(f"#{int(i)} (+{improvements[i]:.4f})" for i in track_indices))
        suptitle = (f'Reconstruction Quality — Top {n_tracks} Elites by Recon Improvement '
                    f'(Finetuned vs. Initial, Phase Aligned)')
    else:
        # Default to a fresh random seed each run so the displayed elites change
        # every time (handy for browsing thesis figure candidates); --seed pins them.
        seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2 ** 32))
        print(f"Track selection seed: {seed}")
        rng           = np.random.default_rng(seed)
        track_indices = rng.choice(len(elite_dataset), size=n_tracks, replace=False)
        suptitle = (f'Reconstruction Quality — {n_tracks} Random Elite Tracks '
                    f'(Finetuned vs. Initial, Phase Aligned)')

    fig, axes = plt.subplots(n_tracks, 3, figsize=(18, n_tracks * 3))
    axes = np.atleast_2d(axes)
    fig.suptitle(suptitle, fontsize=14, y=1.01)

    for row, idx in enumerate(track_indices):
        sample = elite_dataset[idx].unsqueeze(0).to(device)
        orig   = sample[0].float().cpu().numpy()

        recon_ft,   recon_loss_ft   = reconstruct(model,         sample, orig, device, beta, dim_weights)
        recon_init, recon_loss_init = reconstruct(initial_model, sample, orig, device, beta, dim_weights)

        T     = orig.shape[0]
        t     = np.arange(T)

        track_label = (f'Elite {idx} — Recon loss: '
                       f'finetuned {recon_loss_ft:.4f} / initial {recon_loss_init:.4f}')

        for col, (feat, ylim) in enumerate(zip(FEATURE_NAMES, Y_LIMITS)):
            ax = axes[row, col]

            # Original drawn faint/thick as a reference band so the two
            # reconstructions stand out clearly on top of it.
            ax.plot(t, orig[:, col],       color='steelblue',  linewidth=2.5, alpha=0.30)
            ax.plot(t, recon_ft[:, col],   color='darkorange', linewidth=1.4,
                    linestyle='-', alpha=0.95)
            ax.plot(t, recon_init[:, col], color='seagreen',   linewidth=1.4,
                    linestyle='--', alpha=0.95)

            ax.set_ylim(ylim)
            ax.set_xlim(0, T)
            ax.grid(True, alpha=0.25)

            if row == 0 and col == 1:
                ax.text(0.5, 1.08, feat, transform=ax.transAxes, ha='center', va='bottom', fontsize=13, color='black')
                ax.text(0.5, 1.02, track_label, transform=ax.transAxes, ha='center', va='bottom', fontsize=11, color='dimgray')
            elif row == 0:
                ax.set_title(feat, fontsize=13)
            elif col == 1:
                ax.set_title(track_label, fontsize=11, color='dimgray', pad=4)

            if row == n_tracks - 1:
                ax.set_xlabel('Timestep', fontsize=9)

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color='steelblue',  linewidth=2.5, alpha=0.30,     label='Original'),
        Line2D([0], [0], color='darkorange', linewidth=1.5, linestyle='-',  label='Reconstructed (finetuned)'),
        Line2D([0], [0], color='seagreen',   linewidth=1.5, linestyle='--', label='Reconstructed (initial)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    if args.save_fig:
        fig.savefig(args.save_fig, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {args.save_fig}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
