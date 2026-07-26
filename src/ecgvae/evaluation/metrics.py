"""
metrics.py

Evaluation-only metrics for inspecting trained VAEs (as opposed to the
training-time loss terms in training/losses.py).
"""

import torch


def per_dim_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Analytic KL( N(mu, sigma^2) || N(0, I) ) per latent dimension, averaged over the batch."""
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl_per_dim.mean(dim=0)


def prd(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Percent RMS difference between reconstruction and original, averaged over the batch."""
    diff_power = (recon - x).flatten(1).pow(2).sum(dim=1)
    signal_power = x.flatten(1).pow(2).sum(dim=1)
    return 100 * torch.sqrt(diff_power / signal_power).mean()
