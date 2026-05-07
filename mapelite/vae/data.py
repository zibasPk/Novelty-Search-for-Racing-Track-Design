"""Dataset and data-loading utilities for the metrics VAE."""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MetricsDataset(Dataset):
    """Wraps a list of variable-length metric arrays as a PyTorch Dataset."""

    def __init__(self, metrics):
        self.metrics = [torch.tensor(t, dtype=torch.float32) for t in metrics]
        
    def __len__(self):
        return len(self.metrics)

    def __getitem__(self, idx):
        return self.metrics[idx]

def collate_fn(batch):
    lengths = torch.tensor([t.size(0) for t in batch])
    padded = pad_sequence(batch, batch_first=True, padding_value=0.0)
    max_len = padded.size(1)
    mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
    return padded, mask