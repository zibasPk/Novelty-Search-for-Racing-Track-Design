from __future__ import annotations

import numpy as np
import torch

from mapelite.vae import LatentTransform


class MeasureSpace:
    """Single owner of the fitted LatentTransform shared across the pipeline.

    Fits once from the precomputed embedding dataset; EvaluatorMetrics and
    ArchiveVisualizer both hold a reference so they always operate in the
    same whitened space.
    """

    def __init__(self, precomp_embeddings_path: str, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._path = precomp_embeddings_path

        raw = np.load(precomp_embeddings_path)["embeddings"]
        embeddings = torch.tensor(raw, dtype=torch.float32).to(device)
        self._transform = LatentTransform().fit(embeddings)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self._transform.transform(x)

    def get_cleaned_precomp(self) -> np.ndarray:
        """Return the full precomputed dataset after transformation, as numpy."""
        raw = np.load(self._path)["embeddings"]
        emb = torch.tensor(raw, dtype=torch.float32).to(self._device)
        return self.transform(emb).cpu().numpy()

    @property
    def measure_dim(self) -> int:
        """Output dimensionality after dead-dim drop and whitening."""
        return self._transform.n_components_

    def summary(self) -> str:
        return self._transform.summary()
