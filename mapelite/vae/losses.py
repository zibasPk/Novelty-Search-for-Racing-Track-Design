"""VAE loss function for metrics-sequence models."""

import torch

def vae_loss(recon_x, x, mu, log_var, mask=None, beta=0.0, dim_weights=None):
    """Compute the VAE loss: reconstruction + beta * KL divergence.

    Parameters
    ----------
    recon_x : Tensor  - reconstructed sequences.
    x       : Tensor  - original input sequences.
    mu      : Tensor  - latent mean.
    log_var : Tensor  - latent log-variance.
    mask    : BoolTensor | None - padding mask (True = padded).
    beta    : float   - KLD weight (cyclical annealing).
    dim_weights : Tensor | None - dimension wise weighting

    Returns
    -------
    (total_loss, recon_loss, kld_loss)
    """
    if mask is None:
        mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

    inv_mask = (~mask).float()
    squared_error = (recon_x - x) ** 2
    
    if dim_weights is not None:
        squared_error = squared_error * dim_weights
        
    masked_squared_error = squared_error * inv_mask.unsqueeze(-1)

    recon_per_sample = torch.sum(masked_squared_error, dim=[1, 2])
    recon_loss = torch.mean(recon_per_sample)

    kld_per_sample = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kld_loss = torch.mean(kld_per_sample)

    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss

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
            recon_losses.append(torch.tensor(0.0, device=device, requires_grad=True))
            continue

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

    recon_loss = torch.stack(recon_losses).mean()

    log_var_clamped = torch.clamp(log_var, min=-20.0, max=10.0)

    # KLD Loss (Standard)
    kld_per_sample = -0.5 * torch.sum(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp(), dim=1)
    kld_loss = torch.mean(kld_per_sample)

    return recon_loss + beta * kld_loss, recon_loss, kld_loss