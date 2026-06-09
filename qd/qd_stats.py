# qd_stats.py
# Statistics and diagnostics computed over the QD archive each iteration.
# These are pure functions — no runner state required.

import numpy as np
from sklearn.neighbors import NearestNeighbors

from qd.config import INVALID_SCORE


def compute_qd_score(objective_scores: np.ndarray) -> float:
    """Sum of all valid elite fitnesses (QD-Score)."""
    valid = objective_scores[(objective_scores != INVALID_SCORE) & np.isfinite(objective_scores)]
    return float(np.sum(valid)) if len(valid) > 0 else 0.0


def compute_acceptance_rate(new_count: int, sub_count: int, batch_size: int) -> float:
    """Fraction of evaluated candidates accepted (new or improved) into the archive."""
    return (new_count + sub_count) / batch_size if batch_size > 0 else 0.0


def compute_mean_pairwise_dist(measures: np.ndarray, seed=None) -> float:
    """Mean pairwise Euclidean distance among archive members (sampled for speed)."""
    n = len(measures)
    if n < 2:
        return float("nan")
    max_sample = 500
    if n > max_sample:
        rng = np.random.default_rng(seed)
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


def compute_high_quality_coverage(objective_scores: np.ndarray,
                                   threshold: float = 20.0) -> int:
    """Count of archive elites with fitness >= *threshold*."""
    valid = objective_scores[(objective_scores != INVALID_SCORE) & np.isfinite(objective_scores)]
    return int(np.sum(valid >= threshold))


def compute_mean_knn_novelty(measures: np.ndarray, k: int = 5) -> float:
    """Mean k-NN distance among archive members — the NS novelty proxy."""
    if len(measures) < k + 1:
        return float("nan")
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(measures)
    dists, _ = nbrs.kneighbors(measures)
    return float(np.mean(dists[:, 1:]))  # exclude self (column 0)


def compute_fitness_novelty_corr(measures: np.ndarray,
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
