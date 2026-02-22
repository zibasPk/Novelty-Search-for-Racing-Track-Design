import os
import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MetricsTransformerVAE(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, latent_dim=2, n_layers=4, n_heads=4, max_seq_len=200):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=n_layers)
        self.final_projection = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, src_key_padding_mask=None):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(
                (batch_size, 1), dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat(
                [cls_mask, src_key_padding_mask], dim=1)

        x = self.pos_encoding(x)
        x = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask)

        summary = x[:, 0, :]
        mu = self.fc_mu(summary)
        log_var = self.fc_var(summary)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len, src_key_padding_mask=None):
        batch_size = z.shape[0]
        z_projected = self.decoder_input(z).unsqueeze(1)
        queries = self.pos_encoding.pe[:, 1: seq_len + 1, :]
        queries = queries.expand(batch_size, -1, -1)
        x = queries + z_projected
        x = self.transformer_decoder(
            x, src_key_padding_mask=src_key_padding_mask)
        output = self.final_projection(x)
        return output

    def forward(self, x, src_key_padding_mask=None):
        mu, log_var = self.encode(x, src_key_padding_mask)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, x.shape[1], src_key_padding_mask)
        return recon_x, mu, log_var

# ==========================================
# 2. Data Utilities
# ==========================================


class MetricsDataset(Dataset):
    def __init__(self, metrics):
        self.metrics = [torch.tensor(t, dtype=torch.float32) for t in metrics]

    def __len__(self):
        return len(self.metrics)

    def __getitem__(self, idx):
        return self.metrics[idx]


def collate_fn(batch):
    padded_metrics = pad_sequence(batch, batch_first=True, padding_value=0.0)
    mask = (padded_metrics.abs().sum(dim=-1) == 0)
    return padded_metrics, mask


def load_checkpoint(path, device):
    print(f"Loading model from {path}...")
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']

    model = MetricsTransformerVAE(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        n_layers=config.get('n_layers', 4),
        n_heads=config.get('n_heads', 4),
        max_seq_len=config['max_seq_len']
    )

    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded with latent_dim={config['latent_dim']}")
    return model, config['latent_dim']


def preprocess_data(input_data: np.ndarray) -> np.ndarray:
    data = input_data.copy()

    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(
            f"Expected 2D array with 7 columns. With columns: [id, Speed, Steering, Accel, Brake, Gear, distanceToBorder], but got shape {data.shape}")

    # check if data is empty
    if len(data) == 0:
        raise ValueError("Input data is empty. Cannot preprocess an empty dataset.")

    # check if any nans or infs
    if not np.isfinite(data).all():
        raise ValueError("Input data contains NaN or Inf values. Please clean the data before preprocessing.")

    # --- Constants matching Notebook ---
    SPEED_IDX = 0
    STEERING_IDX = 1
    POS_IDX = 2
    MAX_SPEED = 65
    TRACK_WIDTH = 18
    STEERING_THRESHOLD = 0.05

    # 1. Column Selection: ['Speed', 'Steering', 'Track_Angle']
    # Removing [0 (id), 3 (accel), 4 (brake), 5 (gear)]
    data = np.delete(data, [0, 3, 4, 5], axis=1)

    # 2. Normalization
    # Speed
    data[:, SPEED_IDX] = data[:, SPEED_IDX] / MAX_SPEED

    # Position (Clip and Center)
    data[:, POS_IDX] = np.clip(data[:, POS_IDX], 0, TRACK_WIDTH)
    normalized_pos = data[:, POS_IDX] / TRACK_WIDTH
    # Transform to: -0.5 (Left) -> 0.0 (Center) -> +0.5 (Right)
    data[:, POS_IDX] = normalized_pos - 0.5

    # 4. CANONICALIZATION
    # Ensure all tracks start with a Right Turn to stop mirroring issues

    significant_mask = np.abs(data[:, STEERING_IDX]) > STEERING_THRESHOLD
    significant_indices = np.where(significant_mask)[0]

    if len(significant_indices) > 0:
        first_idx = significant_indices[0]
        first_val = data[first_idx, STEERING_IDX]

        # If track starts with Left Turn, MIRROR the whole world
        if first_val < 0:
            data[:, STEERING_IDX] *= -1  # Flip Steering
            # Flip Position (Left side becomes Right side)
            data[:, POS_IDX] *= -1

    return data


def load_model(device, model_path) -> tuple[MetricsTransformerVAE, int]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model, latent_dim = load_checkpoint(model_path, device)

    return model, latent_dim
