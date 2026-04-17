"""Dataset and data-loading utilities for the metrics VAE."""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


class MetricsDataset(Dataset):
    """Wraps a list of variable-length metric arrays as a PyTorch Dataset."""

    def __init__(self, metrics, n_shifts = 1):
        self.metrics = [torch.tensor(t, dtype=torch.float32) for t in metrics]
        self.n_shifts = n_shifts
        
    def __len__(self):
        return len(self.metrics)

    def __getitem__(self, idx):
        x = self.metrics[idx]

        shifted_xs = []
        for _ in range(self.n_shifts):
            shift = torch.randint(0, len(x), (1,)).item()
            shifted_x = torch.roll(x, shift, dims=0)
            shifted_xs.append(shifted_x)
            
        return x, shifted_xs

def collate_fn(batch):
    def process_batch(batch):
        lengths = torch.tensor([t.size(0) for t in batch])
        padded = pad_sequence(batch, batch_first=True, padding_value=0.0)
        max_len = padded.size(1)
        mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
        return padded, mask

    original_batch = [item[0] for item in batch]
    shifted_lists = [item[1] for item in batch]

    # process original batch
    padded, mask = process_batch(original_batch)
    

    padded_shifts_list = []
    mask_shifts_list = []
    n_shifts = len(shifted_lists[0])
    for i in range(n_shifts):
        shift_i_batch = [s[i] for s in shifted_lists]
        padded_i, mask_i = process_batch(shift_i_batch)
        padded_shifts_list.append(padded_i)
        mask_shifts_list.append(mask_i)
    
    return padded, mask, padded_shifts_list, mask_shifts_list