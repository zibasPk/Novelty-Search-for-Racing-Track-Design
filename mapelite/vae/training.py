"""Training loop and helpers for the metrics VAE."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from mapelite.vae.losses import vae_loss
from mapelite.vae.config import TRAINING_CONFIG as _TC

_KLD = _TC["kld"]
_LRS = _TC["lr_schedule"]


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class TrainingConfig:
    """All hyper-parameters consumed by :class:`VAETrainer`."""

    lr: float = _TC["lr"]
    epochs: int = _TC["epochs"]
    patience: int = _TC["patience"]
    min_delta: float = _TC["min_delta"]
    max_grad_norm: float = _TC["max_grad_norm"]

    # Cyclical KLD annealing
    n_cycles: int = _KLD["n_cycles"]
    max_beta: float = _KLD["max_beta"]
    ratio: float = _KLD["ratio"]

    # LR Scheduler
    lr_factor: float = _LRS["factor"]
    lr_patience: int = _LRS["patience"]
    min_lr: float = _LRS["min_lr"]

    # Loss Configuration
    dim_weights: torch.Tensor | None = _TC["dim_weights"]

    @classmethod
    def from_dict(cls, d: dict, dim_weights: torch.Tensor | None = None) -> "TrainingConfig":
        """Build from the legacy ``parameters`` dict format."""
        kld = d.get("kld", {})
        lr_schedule = d.get("lr_schedule", {})
        return cls(
            lr=d.get("lr", _TC["lr"]),
            epochs=d.get("epochs", _TC["epochs"]),
            patience=d.get("patience", _TC["patience"]),
            n_cycles=kld.get("n_cycles", _KLD["n_cycles"]),
            max_beta=kld.get("max_beta", _KLD["max_beta"]),
            ratio=kld.get("ratio", _KLD["ratio"]),
            lr_factor=lr_schedule.get("factor", _LRS["factor"]),
            lr_patience=lr_schedule.get("patience", _LRS["patience"]),
            min_lr=lr_schedule.get("min_lr", _LRS["min_lr"]),
            dim_weights=dim_weights if dim_weights is not None else d.get("dim_weights")
        )


# ── Early Stopping ──────────────────────────────────────────────────────────


class EarlyStopper:
    """Tracks validation loss and stores the best model weights."""

    def __init__(self, patience: int = 5, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.best_model_state = None
        self.best_model_epoch = 0

    def check(self, validation_loss: float, model: torch.nn.Module, epoch: int) -> bool:
        """Return *True* when training should stop."""
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_model_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def load_best_weights(self, model: torch.nn.Module) -> torch.nn.Module:
        """Restore the model to its best recorded state."""
        if self.best_model_state is not None:
            print(
                f"Restoring model to best validation loss from epoch "
                f"{self.best_model_epoch}: "
                f"{self.min_validation_loss:.4f}"
            )
            model.load_state_dict(self.best_model_state)
        return model


# ── Trainer ─────────────────────────────────────────────────────────────────


class VAETrainer:
    """Encapsulates the training state and loop for a VAE model.

    Parameters
    ----------
    model  : nn.Module     – a ``MetricsCircularVAE`` (or compatible).
    config : TrainingConfig – hyper-parameters.
    device : torch.device   – target device.
    """

    def __init__(self, model, config: TrainingConfig, device):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=config.lr_factor,
            patience=config.lr_patience,  
            min_lr=config.min_lr
        )
        self.use_amp = device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.early_stopper = EarlyStopper(
            patience=config.patience, min_delta=config.min_delta
        )
        self.history: dict[str, list | int] = {
            "total_loss": [],   "recon_loss": [],   "kld_loss": [],
            "val_total_loss": [], "val_recon_loss": [], "val_kld_loss": [],
            "beta": [],
        }

    # -- public API -----------------------------------------------------------

    def fit(self, train_loader, val_loader) -> dict[str, list | int]:
        """Run the full training loop. Returns the history dict."""
        cfg = self.config
        epoch_bar = tqdm(range(cfg.epochs), desc="Overall Progress", position=0)

        for epoch in epoch_bar:
            beta = self._compute_beta(epoch)
            train_stats = self._train_epoch(train_loader, beta, epoch)
            val_stats = self._validate_epoch(val_loader, beta)

            self.scheduler.step(val_stats['recon'])
            current_lr = self.optimizer.param_groups[0]['lr']

            self._update_history(train_stats, val_stats, beta)

            if self.early_stopper.check(val_stats["recon"], self.model, epoch):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

            epoch_bar.set_postfix({
                "T_Loss": f"{train_stats['total']:.4f}",
                "V_Loss": f"{val_stats['total']:.4f}",
                "Beta":   f"{beta:.3f}",
            })
            print(
                f"Epoch {epoch + 1}:\n"
                f"  Train — total: {train_stats['total']:.4f} | "
                f"recon: {train_stats['recon']:.4f} | kld: {train_stats['kld']:.4f}\n"
                f"  Val   — total: {val_stats['total']:.4f} | "
                f"recon: {val_stats['recon']:.4f} | kld: {val_stats['kld']:.4f}\n"
                f"  LR: {current_lr:.2e}"
            )

        self.model = self.early_stopper.load_best_weights(self.model)
        self.history['best_epoch'] = self.early_stopper.best_model_epoch
        return self.history

    # -- private helpers ------------------------------------------------------

    def _compute_beta(self, epoch: int) -> float:
        cfg = self.config
        cycle_len = max(1, cfg.epochs // cfg.n_cycles)
        cycle_idx = epoch % cycle_len
        if cycle_idx < cycle_len * cfg.ratio:
            return cfg.max_beta * (cycle_idx / (cycle_len * cfg.ratio))
        return cfg.max_beta

    def _train_epoch(self, loader, beta: float, epoch: int) -> dict:
        self.model.train()
        stats = {"recon": 0.0, "kld": 0.0, "total": 0.0}
        cfg = self.config

        batch_bar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs} [Train]",
            position=1,
            leave=False,
        )

        for data, mask in batch_bar:
            data, mask = data.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()

            with autocast("cuda", enabled=self.use_amp):
                recon_x, mu, log_var = self.model(data, src_key_padding_mask=mask)
                loss, recon, kld = vae_loss(
                    recon_x, data, mu, log_var, mask=mask, beta=beta, dim_weights=cfg.dim_weights
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=cfg.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            stats["recon"] += recon.item()
            stats["kld"] += kld.item()
            stats["total"] += loss.item()
            batch_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon.item():.4f}",
                "kld": f"{kld.item():.4f}"
            })

        n = len(loader)
        return {k: v / n for k, v in stats.items()}

    @torch.no_grad()
    def _validate_epoch(self, loader, beta: float) -> dict:
        self.model.eval()
        stats = {"recon": 0.0, "kld": 0.0, "total": 0.0}
        cfg = self.config

        for data, mask in loader:
            data, mask = data.to(self.device), mask.to(self.device)
            with autocast("cuda", enabled=self.use_amp):
                # Evaluate using mu strictly (no sampling noise)
                mu, log_var = self.model.encode(data, src_key_padding_mask=mask)
                recon_x = self.model.decode(mu, data.shape[1], src_key_padding_mask=mask)
                loss, recon, kld = vae_loss(
                    recon_x, data, mu, log_var, mask=mask, beta=beta, dim_weights=cfg.dim_weights
                )
            stats["recon"] += recon.item()
            stats["kld"] += kld.item()
            stats["total"] += loss.item()

        n = len(loader)
        return {k: v / n for k, v in stats.items()}

    def _update_history(self, train: dict, val: dict, beta: float) -> None:
        self.history["total_loss"].append(train["total"])
        self.history["recon_loss"].append(train["recon"])
        self.history["kld_loss"].append(train["kld"])
        self.history["val_total_loss"].append(val["total"])
        self.history["val_recon_loss"].append(val["recon"])
        self.history["val_kld_loss"].append(val["kld"])
        self.history["beta"].append(beta)