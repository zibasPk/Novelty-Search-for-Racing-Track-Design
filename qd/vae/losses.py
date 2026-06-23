"""VAE loss function for metrics-sequence models."""

import torch

from qd.vae.config import LOSS_CONFIG


def vae_loss(recon_x, x, mu, log_var, mask=None,
             beta=LOSS_CONFIG["beta"], dim_weights=LOSS_CONFIG["dim_weights"]):
    if mask is None:
        mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

    B, T, C = x.shape
    device = x.device

    if dim_weights is not None:
        sqrt_w = torch.sqrt(dim_weights).view(1, 1, C)
        x_scaled = x * sqrt_w
        recon_scaled = recon_x * sqrt_w
    else:
        x_scaled = x
        recon_scaled = recon_x

    recon_losses = []
    valid_lengths = (~mask).sum(dim=1) 

    for i in range(B):
        L = valid_lengths[i].item()
        if L == 0:
            continue

        # Cast to float32 because torch.fft operations don't fully support float16 in AMP
        xi = x_scaled[i, :L, :].float()       
        ri = recon_scaled[i, :L, :].float()   

        sum_sq = xi.pow(2).sum() + ri.pow(2).sum()

        Xi = torch.fft.rfft(xi, dim=0)
        Ri = torch.fft.rfft(ri, dim=0)
        
        CC_c = torch.fft.irfft(Xi * torch.conj(Ri), n=L, dim=0) 
        
        CC = CC_c.sum(dim=-1) 
        max_cc = CC.max()
        
        min_sse = sum_sq - 2 * max_cc
        min_sse = torch.clamp(min_sse, min=0.0)

        recon_losses.append(min_sse)

    # Stack the list into a single tensor and average
    recon_loss = torch.stack(recon_losses).mean()

    log_var_clamped = torch.clamp(
        log_var,
        min=LOSS_CONFIG["log_var_clamp_min"],
        max=LOSS_CONFIG["log_var_clamp_max"],
    )

    # KLD Loss (Standard)
    kld_per_sample = -0.5 * torch.sum(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp(), dim=1)
    kld_loss = torch.mean(kld_per_sample)

    return recon_loss + beta * kld_loss, recon_loss, kld_loss