"""Latent-space health check on the latest archive elites with the finetuned VAE.

This mirrors the "Active Dimension Analysis (train set)" diagnostic, but instead
of the pretraining train set it encodes the *current archive elites* with the
*latest finetuned model* produced by the QD run.

For each elite we recover its raw phenotype sequence from the run's evaluation
buffer (keyed by solution id), preprocess it exactly as the runner does
(``MetricsPreprocessor`` -> ``MetricsDataset`` -> ``collate_fn``), encode it with
the finetuned VAE, and compute the per-dimension KL of the approximate posterior
against the standard-normal prior. Dimensions whose mean KL exceeds
``ACTIVE_THRESHOLD`` nats are considered "active"; the rest have collapsed onto
the prior.

Run from the repository root:

    python -m qd.data_analysis.latent_health_check_elites
    python -m qd.data_analysis.latent_health_check_elites --model data/ns/checkpoints/finetuned_model_1100.pt
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from qd.vae import MetricsVAE, MetricsDataset, MetricsPreprocessor
from qd.vae.data import collate_fn
from qd.vae.config import DATA_CONFIG as _DC

# ── Config ────────────────────────────────────────────────────────────────────
ACTIVE_THRESHOLD = 0.1

DEFAULT_CHECKPOINT_DIR = "data/ns/checkpoints"
DEFAULT_ELITES_FILE    = "data/ns/elites.json"
DEFAULT_BUFFER_FILE    = "data/ns/buffer.json"


def latest_finetuned_model(checkpoint_dir):
    """Return the path of the most recent ``finetuned_model_*.pt`` checkpoint."""
    paths = sorted(glob.glob(os.path.join(checkpoint_dir, "finetuned_model_*.pt")))
    if not paths:
        raise FileNotFoundError(
            f"No finetuned_model_*.pt found in {checkpoint_dir}"
        )
    return paths[-1]


def build_elite_loader(elites_file, buffer_file, batch_size):
    """Build a DataLoader over the elites' preprocessed phenotype sequences.

    The elites carry only their genotype + embedding; their raw phenotype
    telemetry lives in the evaluation buffer, keyed by solution id. We join the
    two, preprocess each sequence (dropping any that fail validation, mirroring
    ``QDRunner._finetune_embedding_model``), and wrap them in the same
    Dataset / collate the runner uses.
    """
    with open(elites_file, "r") as f:
        elites_data = json.load(f)
    with open(buffer_file, "r") as f:
        buffer_data = json.load(f)

    phenotype_by_id = {t["id"]: t.get("phenotype_data") for t in buffer_data["tracks"]}
    elites = elites_data["elites"]
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
        raise ValueError("No valid elite phenotype sequences available to encode.")

    dataset = MetricsDataset(processed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        collate_fn=collate_fn,
    )
    return loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="Path to the finetuned VAE checkpoint "
                             "(default: latest finetuned_model_*.pt under --checkpoint-dir)")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--elites", default=DEFAULT_ELITES_FILE)
    parser.add_argument("--buffer", default=DEFAULT_BUFFER_FILE)
    parser.add_argument("--batch-size", type=int, default=_DC["batch_size"])
    parser.add_argument("--save-fig", default=None,
                        help="Optional path to save the diagnostic figure instead of showing it")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model or latest_finetuned_model(args.checkpoint_dir)
    print(f"Using finetuned model: {model_path}")
    model, latent_dim, _ = MetricsVAE.load_pretrained(model_path, device)

    elite_loader = build_elite_loader(args.elites, args.buffer, args.batch_size)

    # ── Active Dimension Analysis (elite set) ─────────────────────────────────
    model.eval()
    all_mu      = []
    all_log_var = []

    with torch.no_grad():
        for data, mask in tqdm(elite_loader, desc="Encoding elites for KL analysis"):
            data, mask = data.to(device), mask.to(device)
            mu, log_var = model.encode(data, src_key_padding_mask=mask)
            all_mu.append(mu.cpu())
            all_log_var.append(log_var.cpu())

    all_mu      = torch.cat(all_mu, dim=0)
    all_log_var = torch.cat(all_log_var, dim=0)

    kl_per_dim      = -0.5 * (1 + all_log_var - all_mu.pow(2) - all_log_var.exp())
    kl_per_dim_mean = kl_per_dim.mean(dim=0)

    total_kl = kl_per_dim_mean.sum().item()
    kl_np    = kl_per_dim_mean.numpy()

    active_mask = kl_np > ACTIVE_THRESHOLD
    n_active    = active_mask.sum()

    print(f"Elites encoded:         {all_mu.shape[0]}")
    print(f"Total KL:               {total_kl:.2f} nats")
    print(f"KL per dimension (avg): {total_kl / latent_dim:.3f} nats/dim")
    print(f"Active dimensions:      {n_active} / {latent_dim}")
    print(f"Collapsed dimensions:   {latent_dim - n_active} / {latent_dim}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Elite Set — Latent Space Health Check', fontsize=14)

    colors = ['steelblue' if a else 'salmon' for a in active_mask]
    axes[0].bar(range(latent_dim), kl_np, color=colors)
    axes[0].axhline(ACTIVE_THRESHOLD, color='black', linestyle='--', linewidth=1, label=f'Active threshold ({ACTIVE_THRESHOLD})')
    axes[0].axhline(3.0, color='red', linestyle=':', linewidth=1, label='Over-dispersion warning (3.0)')
    axes[0].set_title(f'KL per Dimension — {n_active}/{latent_dim} active')
    axes[0].set_xlabel('Latent Dimension')
    axes[0].set_ylabel('Mean KL (nats)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    sorted_kl = np.sort(kl_np)[::-1]
    axes[1].bar(range(latent_dim), sorted_kl,
                color=['steelblue' if k > ACTIVE_THRESHOLD else 'salmon' for k in sorted_kl])
    axes[1].axhline(ACTIVE_THRESHOLD, color='black', linestyle='--', linewidth=1, label=f'Active threshold ({ACTIVE_THRESHOLD})')
    axes[1].axhline(3.0, color='red', linestyle=':', linewidth=1, label='Over-dispersion warning (3.0)')
    axes[1].set_title('KL per Dimension (Sorted)')
    axes[1].set_xlabel('Rank')
    axes[1].set_ylabel('Mean KL (nats)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if args.save_fig:
        fig.savefig(args.save_fig, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {args.save_fig}")
    else:
        plt.show()

    mu_std = all_mu.std(dim=0)
    print("μ std per dim (sorted):")
    for i, s in enumerate(sorted(mu_std.tolist(), reverse=True)):
        print(f"  rank {i}: std={s:.4f}")


if __name__ == "__main__":
    main()
