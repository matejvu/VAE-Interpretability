"""
vanilla_vae.py

Standard (unconditional) VAE: single Gaussian encoder/decoder pair, with
the original VAE paper's loss (Kingma & Welling, 2013) -- unweighted
reconstruction + KL, no annealing/beta/free-bits.
See BaseVAE (base_vae.py) for the shared reparameterize/forward/sample/
reconstruct logic and the encode/decode/loss_function contract.

Encoder/decoder architecture is intentionally left unbuilt here --
self.encoder / self.decoder are filled in by hand; encode()/decode() below
just need self.encoder to end in a 2 * latent_dim output (split into
mu/logvar) and self.decoder to map latent_dim -> input_length.
"""

import torch
import torch.nn as nn

from ecgvae.models.base_vae import BaseVAE
from ecgvae.training.losses import kl_divergence, reconstruction_loss


class VanillaVAE(BaseVAE):
    def __init__(self, input_length: int, latent_dim: int, hidden_dims: list[int] | None = None):
        super().__init__(input_length, latent_dim)
        self.hidden_dims = hidden_dims or [128, 64, 32]

        self.encoder = nn.Sequential(
            nn.Unflatten(1, (1, input_length)),
            nn.Conv1d(1, self.hidden_dims[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(self.hidden_dims[0], self.hidden_dims[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(self.hidden_dims[1], self.hidden_dims[2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.hidden_dims[2] * (input_length // 32), 2 * latent_dim)  # Output mu and logvar
        )

        # Mirrors the encoder stage-for-stage in reverse (conv/pool -> conv-
        # transpose, each undoing one /2 length-halving), so the 5 halvings
        # in self.encoder become 5 doublings here: input_length // 32 -> input_length.
        # Bridges latent_dim -> the flattened conv-ready size, same role as
        # the encoder's final Linear but inverted.
        self.decoder_input = nn.Linear(latent_dim, self.hidden_dims[2] * (input_length // 32))
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (self.hidden_dims[2], input_length // 32)),
            nn.ConvTranspose1d(self.hidden_dims[2], self.hidden_dims[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(self.hidden_dims[1], self.hidden_dims[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(self.hidden_dims[1], self.hidden_dims[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(self.hidden_dims[0], self.hidden_dims[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(self.hidden_dims[0], 1, kernel_size=4, stride=2, padding=1),
            nn.Flatten(),
            # Reconstruction target is normalized to [0, 1] (see ECGBeatDataset._scale
            # in mitbih.py), so bound the output to match.
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z)
        return self.decoder(h)

    def loss_function(
        self, x: torch.Tensor, outputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Original VAE ELBO: reconstruction + KL to N(0, I), unweighted."""
        recon_loss = reconstruction_loss(outputs["recon"], x)
        kl_loss = kl_divergence(outputs["mu"], outputs["logvar"])
        return {
            "loss": recon_loss + kl_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }