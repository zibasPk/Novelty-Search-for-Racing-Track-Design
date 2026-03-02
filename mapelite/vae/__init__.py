"""mapelite.vae – cleanly separated VAE components.

Public API
----------
Model:
    MetricsTransformerVAE  – transformer VAE architecture.

Data:
    MetricsDataset         – PyTorch dataset for variable-length metrics.
    MetricsDataModule      – dataset + collation + loader factory.

Training:
    VAETrainer             – encapsulated training loop.
    TrainingConfig         – hyper-parameter dataclass.
    EarlyStopper           – patience-based early stopping.

Preprocessing:
    MetricsPreprocessor    – raw telemetry → model-ready features.

Loss:
    vae_loss               – reconstruction + β·KLD loss function.
"""

from mapelite.vae.model import MetricsTransformerVAE
from mapelite.vae.data import MetricsDataset, MetricsDataModule
from mapelite.vae.training import VAETrainer, TrainingConfig, EarlyStopper
from mapelite.vae.preprocessing import MetricsPreprocessor
from mapelite.vae.losses import vae_loss

__all__ = [
    "MetricsTransformerVAE",
    "MetricsDataset",
    "MetricsDataModule",
    "VAETrainer",
    "TrainingConfig",
    "EarlyStopper",
    "MetricsPreprocessor",
    "vae_loss",
]
