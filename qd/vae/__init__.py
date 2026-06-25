"""qd.vae - cleanly separated VAE components.

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

from qd.vae.model import MetricsVAE
from qd.vae.data import MetricsDataset
from qd.vae.training import VAETrainer, TrainingConfig, EarlyStopper
from qd.vae.preprocessing import MetricsPreprocessor
from qd.vae.losses import vae_loss

__all__ = [
    "MetricsVAE",
    "MetricsDataset",
    "VAETrainer",
    "TrainingConfig",
    "EarlyStopper",
    "MetricsPreprocessor",
    "vae_loss",
]