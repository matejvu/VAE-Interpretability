"""
base_vae.py

Abstract base class for all VAE variants (VanillaVAE, BetaVAE,
SemiSupervisedVAE, ...). Defines the encode / reparameterize / decode /
forward / loss_function contract so the trainer (see training/trainer.py)
can work with any subclass interchangeably, regardless of encoder/decoder
architecture or loss composition.

Subclasses must implement:
  - encode(x)                    -> (mu, logvar)
  - decode(z)                    -> reconstruction
  - loss_function(x, outputs)    -> dict of loss terms, must include "loss"

The reparameterization trick, forward pass, prior sampling, and
deterministic reconstruction are identical across variants (same isotropic
Gaussian prior/posterior family), so they're implemented once here instead
of being duplicated in every subclass.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseVAE(nn.Module, ABC):
    def __init__(self, input_length: int, latent_dim: int):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim

    @abstractmethod
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map input x, shape (B, input_length), to (mu, logvar), each (B, latent_dim)."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent z, shape (B, latent_dim), to a reconstruction, shape (B, input_length)."""
        raise NotImplementedError

    

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, sigma^2) via the reparameterization trick (differentiable)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | None = None) -> torch.Tensor:
        """Draw num_samples from the generative model: z ~ N(0, I), then decode."""
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction: encode to mu (skip sampling noise), then decode."""
        mu, _ = self.encode(x)
        return self.decode(mu)