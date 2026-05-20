"""mapelite.vae - cleanly separated VAE components.

Public API
----------
Model:
    MetricsVAE  - circular-convolution VAE with DFT frequency pooling.

Data:
    MetricsDataset         - PyTorch dataset for variable-length metrics.
Training:
    VAETrainer             - encapsulated training loop.
    TrainingConfig         - hyper-parameter dataclass.
    EarlyStopper           - patience-based early stopping.

Preprocessing:
    MetricsPreprocessor    - raw telemetry -> model-ready features.

Loss:
    vae_loss               - shift-invariant reconstruction + beta*KLD loss.
"""

from mapelite.vae.model import MetricsVAE
from mapelite.vae.data import MetricsDataset
from mapelite.vae.training import VAETrainer, TrainingConfig, EarlyStopper
from mapelite.vae.preprocessing import MetricsPreprocessor
from mapelite.vae.losses import vae_loss
from mapelite.vae.latent_transform import LatentTransform

__all__ = [
    "MetricsVAE",
    "MetricsDataset",
    "VAETrainer",
    "TrainingConfig",
    "EarlyStopper",
    "MetricsPreprocessor",
    "LatentTransform",
    "vae_loss",
]