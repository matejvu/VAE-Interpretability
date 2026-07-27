"""
fc_vanilla_vae.py

Same vanilla ELBO as VanillaVAE (see vanilla_vae.py), but with a purely
fully-connected encoder/decoder -- no Conv1d/pooling. Comparison baseline
for whether the conv inductive bias (local, shift-invariant filters) is
actually pulling weight on these 256-sample beat windows.

hidden_dims defaults to [144, 72, 36], sized to land around ~100k total
params -- FC layers are dense (every unit connects to every input unit),
so matching VanillaVAE's channel counts directly would run several times
larger; these are picked smaller on purpose for a comparable model size.
"""

import torch
import torch.nn as nn

from ecgvae.models.base_vae import BaseVAE
from ecgvae.training.losses import kl_divergence, reconstruction_loss


class FCVanillaVAE(BaseVAE):
    def __init__(self, input_length: int, latent_dim: int, hidden_dims: list[int] | None = None):
        super().__init__(input_length, latent_dim)
        self.hidden_dims = hidden_dims or [144, 72, 36]
        h0, h1, h2 = self.hidden_dims

        self.encoder = nn.Sequential(
            nn.Linear(input_length, h0),
            nn.ReLU(),
            nn.Linear(h0, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 2 * latent_dim),  # output mu and logvar
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, h0),
            nn.ReLU(),
            nn.Linear(h0, input_length),
            # Reconstruction target is normalized to [0, 1] (see ECGBeatDataset._scale
            # in mitbih.py), so bound the output to match.
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def loss_function(
        self, x: torch.Tensor, outputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Same original-VAE ELBO as VanillaVAE (see vanilla_vae.py) --
        unweighted reconstruction + KL. Only the architecture differs."""
        recon_loss = reconstruction_loss(outputs["recon"], x)
        kl_loss = kl_divergence(outputs["mu"], outputs["logvar"])
        return {
            "loss": recon_loss + kl_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
