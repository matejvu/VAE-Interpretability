"""
losses.py

Reusable loss-term primitives for the ecgvae VAE variants. Each variant's
`loss_function` (see base_vae.py's default, and any subclass override)
composes these -- kept separate so different compositions (unweighted
ELBO, beta-weighted KL, + a classification term, ...) reuse the same
reconstruction/KL math instead of each reimplementing it.

All losses are summed over non-batch dimensions then divided by batch
size -- i.e. "total loss for one example, averaged across the batch".
"""

import torch
import torch.nn.functional as F


def reconstruction_loss(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sum-of-squared-error reconstruction loss, averaged over the batch."""
    return F.mse_loss(recon, x, reduction="sum") / x.size(0)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Analytic KL( N(mu, sigma^2) || N(0, I) ), averaged over the batch."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
