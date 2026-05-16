from __future__ import annotations

from pathlib import Path

import torch


class LatentTransform:
    """Drops dead VAE latent dimensions, then PCA-whitens the rest.

    Parameters
    ----------
    variance_threshold : float
        Per-dim variance below which a latent dimension is considered
        "dead" and dropped. Computed on the training embeddings.
    eps : float
        Small constant added to eigenvalues before inverse-sqrt to keep
        the whitening numerically stable.
    explained_variance_ratio : float | None
        If given (e.g. ``0.99``), keep only the top PCA components that
        together explain this fraction of variance. ``None`` keeps all
        components of the surviving dimensions.
    """

    def __init__(
        self,
        variance_threshold: float = 0.1,
        eps: float = 1e-5,
        explained_variance_ratio: float | None = None,
    ):
        self.variance_threshold = variance_threshold
        self.eps = eps
        self.explained_variance_ratio = explained_variance_ratio

        # Set by .fit()
        self.active_dims_: torch.Tensor | None = None   # indices kept after drop
        self.mean_: torch.Tensor | None = None          # mean of active dims
        self.whitening_: torch.Tensor | None = None     # [D_active, n_components]
        self.eigenvalues_: torch.Tensor | None = None   # variance per PC (pre-whitening)
        self.n_components_: int | None = None

    # ── fit / transform ──────────────────────────────────────────────────────

    def fit(self, embeddings: torch.Tensor) -> "LatentTransform":
        """Estimate active dims and the whitening matrix from training μ.

        Parameters
        ----------
        embeddings : torch.Tensor
            Shape ``[N, latent_dim]``. The μ outputs of the encoder over
            the full training set (use ``mu``, not the sampled ``z``).
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected [N, D], got {tuple(embeddings.shape)}")
        if embeddings.shape[0] < 2:
            raise ValueError("Need at least 2 samples to estimate variance.")

        # Work in float64 on CPU for numerical headroom; cast back at the end.
        X_all = embeddings.detach().to(torch.float64).cpu()

        # 1. Dead-dim drop ----------------------------------------------------
        per_dim_var = X_all.var(dim=0, unbiased=True)
        active = (per_dim_var > self.variance_threshold).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            raise RuntimeError(
                "All latent dimensions are dead (variance below threshold). "
                "The encoder appears to have collapsed — check β and KLD."
            )

        X = X_all[:, active]                                # [N, D_active]

        # 2. PCA whitening via SVD of centred data ---------------------------
        mean = X.mean(dim=0)
        Xc = X - mean
        N = Xc.shape[0]

        # Xc = U S V^T  →  eigenvalues of Cov(Xc) are S^2 / (N-1)
        _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        eigvals = (S ** 2) / max(N - 1, 1)                  # [k]

        # Optional truncation to a fraction of explained variance
        if self.explained_variance_ratio is not None:
            total = eigvals.sum().clamp_min(1e-12)
            cumulative = torch.cumsum(eigvals / total, dim=0)
            k = int(torch.searchsorted(
                cumulative, torch.tensor(self.explained_variance_ratio, dtype=cumulative.dtype)
            ).item()) + 1
            k = min(k, eigvals.numel())
            Vh, eigvals = Vh[:k], eigvals[:k]

        # Whitening: z = (x - mean) @ W,  with W = V · diag(1 / sqrt(λ + eps))
        whitening = Vh.T / torch.sqrt(eigvals + self.eps)   # [D_active, n_components]

        # Persist as float32 for downstream use
        self.active_dims_ = active.long()
        self.mean_ = mean.float()
        self.whitening_ = whitening.float()
        self.eigenvalues_ = eigvals.float()
        self.n_components_ = int(whitening.shape[1])
        return self

    def transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply dead-dim drop + whitening. ``[N, D] → [N, n_components_]``.

        Also accepts a single embedding of shape ``[D]`` and returns ``[n_components_]``.
        """
        if self.active_dims_ is None:
            raise RuntimeError("LatentTransform must be .fit() before .transform()")

        single = embeddings.ndim == 1
        if single:
            embeddings = embeddings.unsqueeze(0)

        device, dtype = embeddings.device, embeddings.dtype
        active = self.active_dims_.to(device)
        mean = self.mean_.to(device=device, dtype=dtype)
        W = self.whitening_.to(device=device, dtype=dtype)

        z = (embeddings[:, active] - mean) @ W
        return z.squeeze(0) if single else z

    def fit_transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.fit(embeddings).transform(embeddings)

    # ── diagnostics ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        if self.active_dims_ is None:
            return "LatentTransform(unfitted)"
        total_var = self.eigenvalues_.sum().item()
        return (
            f"LatentTransform(active_dims={self.active_dims_.numel()}, "
            f"n_components={self.n_components_}, "
            f"explained_variance≈{total_var:.4f})"
        )