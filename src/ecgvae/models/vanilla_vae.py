"""
vanilla_vae.py

Standard (unconditional) VAE: single Gaussian encoder/decoder pair, no
extra terms beyond the vanilla ELBO (reconstruction + unweighted KL).
See BaseVAE (base_vae.py) for the shared reparameterize/forward/sample/
reconstruct logic and the encode/decode/loss_function contract.

Encoder/decoder architecture is intentionally left unbuilt here --
self.encoder / self.decoder are filled in by hand; encode()/decode() below
just need self.encoder to end in a 2 * latent_dim output (split into
mu/logvar) and self.decoder to map latent_dim -> input_length.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ecgvae.models.base_vae import BaseVAE


class VanillaVAE(BaseVAE):
    def __init__(self, input_length: int, latent_dim: int, hidden_dims: list[int] | None = None):
        super().__init__(input_length, latent_dim)
        self.hidden_dims = hidden_dims or [128, 64, 32]

        # TODO: build the actual encoder/decoder networks by hand.
        # self.encoder: (B, input_length) -> (B, 2 * latent_dim), split by encode() below.
        # self.decoder: (B, latent_dim) -> (B, input_length).
        self.encoder = None
        self.decoder = None

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def loss_function(
        self, x: torch.Tensor, outputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Standard ELBO: reconstruction + unweighted KL to N(0, I), both averaged over the batch."""
        recon, mu, logvar = outputs["recon"], outputs["mu"], outputs["logvar"]

        recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        return {
            "loss": recon_loss + kl_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }