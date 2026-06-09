"""Circular-CNN VAE for track-metrics sequences."""

import os
import math

import torch
import torch.nn as nn

from qd.vae.config import MODEL_CONFIG, MODEL_INIT


def normalized_dft_pool(h, lengths, K=MODEL_CONFIG["freq_bins"]):
    """DFT power pooling at K normalized frequencies (cycles/lap).

    Args:
        h:       [B, C, T] — features, padded
        lengths: [B]        — actual sequence length per sample
    Returns:     [B, C, K]  — power at K normalized frequencies
    """
    B, C, T = h.shape
    device  = h.device

    t     = torch.arange(T, device=device, dtype=torch.float32)
    k     = torch.arange(0, K, device=device, dtype=torch.float32)
    L     = lengths.float().clamp(min=1).view(-1, 1, 1)

    phase = 2 * math.pi * k.view(1, -1, 1) * t.view(1, 1, -1) / L

    valid = (t.view(1, -1) < lengths.view(-1, 1)).unsqueeze(1).float()
    cos_b = phase.cos() * valid
    sin_b = phase.sin() * valid

    real  = torch.einsum('bct,bkt->bck', h, cos_b) / L
    imag  = -torch.einsum('bct,bkt->bck', h, sin_b) / L

    return real.pow(2) + imag.pow(2)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MODEL_CONFIG["max_seq_len"]):
        super().__init__()
        
        # Create a matrix of [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # The denominator term (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Fill sine for even indices, cosine for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # register_buffer ensures this is saved with the model but not trained as a weight
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Hidden_Dim]
        # Add the encoding to the input
        return x + self.pe[:, :x.size(1), :]

class CircularConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, dilation=dilation
        )

    def forward(self, x, lengths=None):
        B, C, T = x.shape
    
        if lengths is not None:
            assert (lengths > 0).all(), "CircularConv1d requires all sequence lengths > 0"
            # (lengths[b] - pad + k) % lengths[b] for k in 0..pad-1
            # When lengths >= pad: identical to the old "slice from tail" behaviour
            # When lengths < pad:  modulo causes the sequence to repeat, e.g.
            #   length=3, pad=5  →  indices [1, 2, 0, 1, 2] instead of [0, 0, 0, 1, 2]
            start = lengths.unsqueeze(1) - self.pad                         # [B, 1]
            seq   = start + torch.arange(self.pad, device=x.device)        # [B, pad]
            safe  = seq % lengths.unsqueeze(1)                             # [B, pad]
            wrap  = torch.gather(
                x, dim=2,
                index=safe.unsqueeze(1).expand(-1, C, -1)                  # [B, C, pad]
            )
        else:
            # Unmasked / inference path — same fix so it holds when T < pad too.
            # Without this, x[:, :, -pad:] silently returns fewer than pad elements
            # when T < pad, making the cat the wrong length and breaking the conv.
            seq  = torch.arange(T - self.pad, T, device=x.device) % T     # [pad]
            wrap = torch.gather(
                x, dim=2,
                index=seq.view(1, 1, -1).expand(B, C, -1)                 # [B, C, pad]
            )
    
        return self.conv(torch.cat([wrap, x], dim=2))
class ChannelLayerNorm(nn.Module):
    """Applies LayerNorm across the channel dimension for (B, C, T) tensors."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # Transpose to [B, T, C], normalize, transpose back to [B, C, T]
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)
        
class CircularResBlock(nn.Module):
    def __init__(self, channels, kernel_size=MODEL_CONFIG["kernel_size"], dilation=1):
        super().__init__()
        self.conv1 = CircularConv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = CircularConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1  = ChannelLayerNorm(channels)
        self.norm2  = ChannelLayerNorm(channels)
        self.act    = nn.GELU()

    def forward(self, x, lengths=None, valid=None):   # <-- accept valid mask
        if valid is not None:
            x = x * valid
            
        residual = x
        x = self.act(self.norm1(self.conv1(x, lengths)))
        
        # Since ChannelLayerNorm could create non zero values in padded areas
        if valid is not None:
            x = x * valid # reapply mask
            
        x = self.norm2(self.conv2(x, lengths))
        
        if valid is not None:
            x = x * valid
        return x + residual


class CircularCNNDecoder(nn.Module):
    """
    Mirrors the encoder: broadcast z across time, then refine
    with circular ResBlocks in reverse dilation order.
    
    Shift-equivariant by construction — consistent with the encoder.
    """
    def __init__(self, latent_dim, hidden_dim, output_dim,
                 n_layers=MODEL_CONFIG["n_layers"],
                 kernel_size=MODEL_CONFIG["kernel_size"],
                 max_seq_len=MODEL_CONFIG["max_seq_len"]):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)

        self.pe = SinusoidalPositionalEncoding(hidden_dim, max_len=max_seq_len)
        
        # Reverse dilation order: 8,4,2,1 — coarse-to-fine refinement
        self.conv_blocks = nn.ModuleList([
            CircularResBlock(
                hidden_dim, 
                kernel_size=kernel_size, 
                dilation=2 ** (n_layers - 1 - i)
            )
            for i in range(n_layers)
        ])
        self.final_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, seq_len, src_key_padding_mask=None):
        h = self.fc(z)
        h = h.unsqueeze(-1).expand(-1, -1, seq_len)

        # Inject Time Signal
        h = h.transpose(1, 2)  # To [B, T, C] for PE
        h = self.pe(h)         # Add PE
        h = h.transpose(1, 2)  # Back to [B, C, T] for Convs
    
        if src_key_padding_mask is not None:
            valid   = (~src_key_padding_mask).unsqueeze(1).float()
            lengths = (~src_key_padding_mask).sum(dim=1)             # <-- NEW
        else:
            lengths = None
            valid = None
    
        for block in self.conv_blocks:
            h = block(h, lengths, valid)
    
        h = h.transpose(1, 2)
        raw_output = self.final_projection(h)

        # Slice the tensor by feature and apply the correct physical bounds
        speed = torch.sigmoid(raw_output[:, :, 0:1])
        steer = torch.tanh(raw_output[:, :, 1:2]) * 0.5
        pos   = torch.tanh(raw_output[:, :, 2:3]) * 0.5

        final_output = torch.cat([speed, steer, pos], dim=2)
        
        if src_key_padding_mask is not None:
            final_output = final_output * valid.transpose(1, 2)

        return final_output

class MetricsVAE(nn.Module):
    """
    Encoder:  Linear projection → stacked CircularResBlocks → DFT power pool
              → fc_mu / fc_var
              Shift invariance is guaranteed by construction: no contrastive
              loss needed.

    Decoder:  Circular CNN decoder. Reconstructs in the time domain given z
              and seq_len.
    """
    def __init__(self,
                 input_dim=MODEL_CONFIG["input_dim"],
                 hidden_dim=MODEL_CONFIG["hidden_dim"],
                 latent_dim=MODEL_CONFIG["latent_dim"],
                 n_layers=MODEL_CONFIG["n_layers"],
                 kernel_size=MODEL_CONFIG["kernel_size"],
                 max_seq_len=MODEL_CONFIG["max_seq_len"],
                 freq_bins=MODEL_CONFIG["freq_bins"]):
        super().__init__()

        self.hidden_dim  = hidden_dim
        self.latent_dim  = latent_dim
        self.n_layers    = n_layers
        self.max_seq_len = max_seq_len
        self.freq_bins   = freq_bins

        # ── Encoder ──────────────────────────────────────────────────────────
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.conv_blocks = nn.ModuleList([
            CircularResBlock(hidden_dim, kernel_size=kernel_size, dilation=2**i)
            for i in range(n_layers)
        ])

        # DFT pool collapses T → freq_bins power values per channel
        pool_dim = hidden_dim * freq_bins
        self.fc_mu  = nn.Linear(pool_dim, latent_dim)
        self.fc_var = nn.Linear(pool_dim, latent_dim)

        # This prevents the massive KLD explosion in Epoch 1
        nn.init.constant_(self.fc_var.bias, MODEL_INIT["fc_var_bias_init"])
        nn.init.orthogonal_(self.fc_var.weight, gain=MODEL_INIT["fc_var_weight_gain"])

        # ── Decoder ──
        self.decoder = CircularCNNDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            n_layers=n_layers,
            kernel_size=kernel_size,
            max_seq_len=max_seq_len,
        )

    @classmethod
    def load_pretrained(cls, path, device):
        """Load a checkpoint saved with ``torch.save({'config': …, 'state_dict': …, 'parameters': …})``."""
        import os
        import torch
        
        print(f"Loading model from {path}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        
        # Load the training parameters/hyperparameters saved in the checkpoint
        # Using .get() ensures it won't crash if loading an older checkpoint without them
        params = checkpoint.get("parameters", {})

        model = cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            n_layers=config.get("n_layers", MODEL_CONFIG["n_layers"]),
            max_seq_len=config.get("max_seq_len", MODEL_CONFIG["max_seq_len"]),
            kernel_size=config.get("kernel_size", MODEL_CONFIG["kernel_size"]),
            freq_bins=config.get("freq_bins", MODEL_CONFIG["freq_bins"]),
        )
        
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()

        print(f"Model loaded with latent_dim={config['latent_dim']}")
        
        return model, config["latent_dim"], params

    # ── Encoder ──────────────────────────────────────────────────────────────
    def encode(self, x, src_key_padding_mask=None):
        h = self.input_projection(x)
        h = h.transpose(1, 2)
        B, C, T = h.shape

        if src_key_padding_mask is not None:
            valid        = (~src_key_padding_mask).unsqueeze(1).float()
            lengths      = (~src_key_padding_mask).sum(dim=1)
            pool_lengths = lengths
        else:
            valid        = None
            lengths      = None
            pool_lengths = torch.full((B,), T, device=h.device, dtype=torch.long)

        for block in self.conv_blocks:
            h = block(h, lengths, valid)

        power = normalized_dft_pool(h, pool_lengths, K=self.freq_bins)
        power = power.flatten(1)

        return self.fc_mu(power), self.fc_var(power)

    # ── Decoder ──────────────────────────────────────────────────────────────
    def decode(self, z, seq_len, src_key_padding_mask=None):
        return self.decoder(z, seq_len, src_key_padding_mask)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def forward(self, x, src_key_padding_mask=None):
        mu, log_var = self.encode(x, src_key_padding_mask)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decode(z, x.shape[1], src_key_padding_mask)
        return recon, mu, log_var