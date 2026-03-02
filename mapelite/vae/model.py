"""Transformer-based VAE for track-metrics sequences."""

import os
import math

import torch
import torch.nn as nn
import numpy as np


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


class MetricsTransformerVAE(nn.Module):
    """Transformer VAE that encodes variable-length metric sequences
    into a fixed-size latent space and decodes them back.
    """

    def __init__(
        self,
        input_dim=4,
        hidden_dim=64,
        latent_dim=2,
        n_layers=4,
        n_heads=4,
        max_seq_len=200,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=n_layers
        )
        self.final_projection = nn.Linear(hidden_dim, input_dim)

    @classmethod
    def load_pretrained(cls, path, device):
        """Load a checkpoint saved with ``torch.save({'config': …, 'state_dict': …})``."""
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
            n_heads=config.get("n_heads", 4),
            max_seq_len=config["max_seq_len"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        print(f"Model loaded with latent_dim={config['latent_dim']}")
        return model, config["latent_dim"]

    # -- core methods ---------------------------------------------------------

    def encode(self, x, src_key_padding_mask=None):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(
                (batch_size, 1), dtype=torch.bool, device=x.device
            )
            src_key_padding_mask = torch.cat(
                [cls_mask, src_key_padding_mask], dim=1
            )

        x = self.pos_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

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
        queries = self.pos_encoding.pe[:, 1 : seq_len + 1, :]
        queries = queries.expand(batch_size, -1, -1)
        x = queries + z_projected
        x = self.transformer_decoder(x, src_key_padding_mask=src_key_padding_mask)
        output = self.final_projection(x)
        return output

    def forward(self, x, src_key_padding_mask=None):
        mu, log_var = self.encode(x, src_key_padding_mask)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, x.shape[1], src_key_padding_mask)
        return recon_x, mu, log_var
