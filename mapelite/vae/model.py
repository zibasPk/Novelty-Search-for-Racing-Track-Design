"""Circular-CNN VAE for track-metrics sequences."""

import os
import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class CircularConv1d(nn.Module):
    """Conv1d with circular wrap-around padding and preserved sequence length."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x, lengths=None):
        if lengths is not None:
            B, C, T = x.shape
            start_idx = lengths.unsqueeze(1) - self.pad
            seq_idx = start_idx + torch.arange(self.pad, device=x.device)
            valid_mask = (seq_idx >= 0).unsqueeze(1)
            safe_idx = seq_idx.clamp(min=0)
            safe_idx_expanded = safe_idx.unsqueeze(1).expand(-1, C, -1)
            wrap = torch.gather(x, dim=2, index=safe_idx_expanded)
            wrap = wrap * valid_mask.float()
        else:
            wrap = x[:, :, -self.pad:]

        x = torch.cat([wrap, x], dim=2)
        return self.conv(x)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


class CircularResBlock(nn.Module):
    """Residual block with two circular convolutions."""

    def __init__(self, channels, kernel_size=7, dilation=1):
        super().__init__()
        self.conv1 = CircularConv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = CircularConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = ChannelLayerNorm(channels)
        self.norm2 = ChannelLayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x, lengths=None, valid=None):
        residual = x
        x = self.act(self.norm1(self.conv1(x, lengths)))
        if valid is not None:
            x = x * valid
        x = self.norm2(self.conv2(x, lengths))
        return self.act(x + residual)


class CircularCNNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, n_layers=4, kernel_size=7):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.pe = SinusoidalPositionalEncoding(hidden_dim, max_len=5000)

        self.conv_blocks = nn.ModuleList(
            [
                CircularResBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2 ** (n_layers - 1 - i),
                )
                for i in range(n_layers)
            ]
        )
        self.final_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, seq_len, src_key_padding_mask=None):
        h = self.fc(z)
        h = h.unsqueeze(-1).expand(-1, -1, seq_len)

        h = h.transpose(1, 2)
        h = self.pe(h)
        h = h.transpose(1, 2)

        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).unsqueeze(1).float()
            lengths = (~src_key_padding_mask).sum(dim=1)
        else:
            lengths = None
            valid = None

        for block in self.conv_blocks:
            if valid is not None:
                h = h * valid
            h = block(h, lengths, valid)

        if valid is not None:
            h = h * valid

        h = h.transpose(1, 2)
        return self.final_projection(h)


class MetricsTransformerVAE(nn.Module):
    """Shift-invariant circular CNN VAE for variable-length metrics."""

    def __init__(
        self,
        input_dim=3,
        hidden_dim=128,
        latent_dim=32,
        n_layers=4,
        n_heads=8,
        max_seq_len=5000,
        kernel_size=7,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.kernel_size = kernel_size

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.conv_blocks = nn.ModuleList(
            [
                CircularResBlock(hidden_dim, kernel_size=kernel_size, dilation=2**i)
                for i in range(n_layers)
            ]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        nn.init.constant_(self.fc_var.bias, -2.0)
        nn.init.orthogonal_(self.fc_var.weight, gain=0.01)

        self.decoder = CircularCNNDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            n_layers=n_layers,
            kernel_size=kernel_size,
        )

    @classmethod
    def load_pretrained(cls, path, device):
        print(f"Loading model from {path}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]

        model = cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            n_layers=config.get("n_layers", 4),
            n_heads=config.get("n_heads", 8),
            max_seq_len=config.get("max_seq_len", 5000),
            kernel_size=config.get("kernel_size", 7),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        print(f"Model loaded with latent_dim={config['latent_dim']}")
        return model, config["latent_dim"]

    def encode(self, x, src_key_padding_mask=None):
        h = self.input_projection(x)
        h = h.transpose(1, 2)

        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).unsqueeze(1).float()
            lengths = (~src_key_padding_mask).sum(dim=1)
        else:
            lengths = None
            valid = None

        for block in self.conv_blocks:
            if valid is not None:
                h = h * valid
            h = block(h, lengths, valid)

        if valid is not None:
            lengths_f = lengths.float().unsqueeze(1)
            h = (h * valid).sum(dim=2) / (lengths_f + 1e-9)
        else:
            h = h.mean(dim=2)

        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z, seq_len, src_key_padding_mask=None):
        return self.decoder(z, seq_len, src_key_padding_mask)

    def forward(self, x, src_key_padding_mask=None):
        mu, log_var = self.encode(x, src_key_padding_mask)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, x.shape[1], src_key_padding_mask)
        return recon_x, mu, log_var

