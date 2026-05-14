"""VAE loss function for metrics-sequence models."""

import torch

def shift_invariant_vae_loss_fn(recon_x, x, mu, log_var, mask=None, beta=0, dim_weights=None):
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
        
        min_mse = sum_sq - 2 * max_cc
        min_mse = torch.clamp(min_mse, min=0.0)

        recon_losses.append(min_mse)

    # Stack the list into a single tensor and average
    recon_loss = torch.stack(recon_losses).mean()

    log_var_clamped = torch.clamp(log_var, min=-20.0, max=10.0)

    # KLD Loss (Standard)
    kld_per_sample = -0.5 * torch.sum(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp(), dim=1)
    kld_loss = torch.mean(kld_per_sample)

    return recon_loss + beta * kld_loss, recon_loss, kld_loss