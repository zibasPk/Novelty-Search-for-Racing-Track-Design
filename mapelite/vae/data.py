"""Dataset and data-loading utilities for the metrics VAE."""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


class MetricsDataset(Dataset):
    """Wraps a list of variable-length metric arrays as a PyTorch Dataset."""

    def __init__(self, metrics):
        self.metrics = [torch.tensor(t, dtype=torch.float32) for t in metrics]

    def __len__(self):
        return len(self.metrics)

    def __getitem__(self, idx):
        return self.metrics[idx]


class MetricsDataModule:
    """Groups dataset creation, collation, and data-loader construction.

    Parameters
    ----------
    metrics    : list of array-like – one entry per track.
    batch_size : int
    val_split  : float – fraction reserved for validation.
    """

    def __init__(self, metrics, batch_size: int = 32, val_split: float = 0.2):
        self.batch_size = batch_size
        train, val = train_test_split(metrics, test_size=val_split)
        self.train_ds = MetricsDataset(train)
        self.val_ds = MetricsDataset(val)

    # -- collation ------------------------------------------------------------

    @staticmethod
    def collate_fn(batch):
        """Pad variable-length sequences and produce a boolean mask."""
        lengths = torch.tensor([t.size(0) for t in batch])
        padded = pad_sequence(batch, batch_first=True, padding_value=0.0)
        max_len = padded.size(1)
        mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
        return padded, mask

    # -- loaders --------------------------------------------------------------

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_loader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
