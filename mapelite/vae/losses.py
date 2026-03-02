"""VAE loss function for the metrics transformer model."""

import torch


def vae_loss(recon_x, x, mu, log_var, mask=None, beta=0.001):
    """Compute the VAE loss: reconstruction + β · KL divergence.

    Parameters
    ----------
    recon_x : Tensor  – reconstructed sequences.
    x       : Tensor  – original input sequences.
    mu      : Tensor  – latent mean.
    log_var : Tensor  – latent log-variance.
    mask    : BoolTensor | None – padding mask (True = padded).
    beta    : float   – KLD weight (cyclical annealing).

    Returns
    -------
    (total_loss, recon_loss, kld_loss)
    """
    if mask is None:
        mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

    inv_mask = (~mask).float()
    squared_error = (recon_x - x) ** 2
    masked_squared_error = squared_error * inv_mask.unsqueeze(-1)

    recon_loss = torch.mean(torch.sum(masked_squared_error, dim=[1, 2]))
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    )

    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss
