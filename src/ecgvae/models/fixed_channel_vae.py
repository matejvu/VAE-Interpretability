"""
fixed_channel_vae.py

Same conv/deconv architecture as FBVanillaVAE (see fb_vanilla_vae.py), but
with a different loss: the Burgess et al. (2018) "controlled capacity
increase" objective, reconstruction + gamma * |KL - C|, instead of
reconstruction + KL. `gamma`/`C` are real per-run hyperparameters (from
your YAML config); the architecture, by contrast, is meant to stay fixed
across runs -- BASE_CHANNELS/N_CONV_LAYERS/LATENT_DIM below are the single
place to change those defaults if needed.

See BaseVAE (base_vae.py) for the shared reparameterize/forward/sample/
reconstruct logic and the encode/decode/loss_function contract.
"""

import torch
import torch.nn as nn

from ecgvae.models.base_vae import BaseVAE
from ecgvae.training.losses import kl_divergence, reconstruction_loss

BASE_CHANNELS = 16
N_CONV_LAYERS = 3
LATENT_DIM = 16


class FixedChannelVAE(BaseVAE):
    def __init__(
        self,
        input_length: int,
        latent_dim: int = LATENT_DIM,
        gamma: float | None = None,
        C: float | None = None,
        base_channels: int = BASE_CHANNELS,
        n_conv_layers: int = N_CONV_LAYERS,
    ):
        super().__init__(input_length, latent_dim)

        if gamma is None or C is None:
            raise ValueError("FixedChannelVAE requires both gamma and C (no default -- set them in your config).")
        self.gamma = gamma
        self.C = C

        # --- channel schedule: 1 -> 16 -> 32 -> 64, stride 2 each (base_channels=16, n_conv_layers=3) ---
        channels = [1] + [base_channels * (2 ** min(i, 2)) for i in range(n_conv_layers)]

        conv_layers = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
        conv_stack = nn.Sequential(*conv_layers)

        # Infer flattened bottleneck size generically (robust to input_length
        # or n_conv_layers changes) rather than hardcoding it.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            dummy_out = conv_stack(dummy)
            self._decoder_channels = dummy_out.shape[1]
            self._decoder_length = dummy_out.shape[2]
            flat_dim = dummy_out.numel()

        self.encoder = nn.Sequential(
            # x arrives as (B, input_length); conv_stack (and the dummy shape
            # inference above) expects (B, 1, input_length) -- add the
            # channel dim here, not inside conv_stack itself.
            nn.Unflatten(1, (1, input_length)),
            conv_stack,
            nn.Flatten(),
            nn.Linear(flat_dim, 2 * latent_dim),  # split into mu/logvar in encode()
        )

        # --- decoder: mirror of the encoder ---
        self.decoder_input = nn.Linear(latent_dim, flat_dim)

        rev_channels = channels[::-1]
        deconv_layers = []
        for idx, (in_ch, out_ch) in enumerate(zip(rev_channels[:-1], rev_channels[1:])):
            is_last = idx == len(rev_channels) - 2
            deconv_layers.append(
                nn.ConvTranspose1d(
                    in_ch, out_ch, kernel_size=7, stride=2, padding=3,
                    output_padding=1, bias=is_last,  # bias only needed if no BN follows
                )
            )
            if not is_last:
                deconv_layers += [nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True)]
            else:
                # Final layer: no BN (would distort the output range right
                # before the bounding activation). Flatten drops the size-1
                # channel dim (B, 1, input_length) -> (B, input_length) to
                # match x's shape -- without it, loss_function's MSE against
                # x silently broadcasts instead of comparing element-wise.
                # Sigmoid assumes your data is min-max scaled to [0, 1] --
                # swap for Tanh (+ [-1, 1] data) or drop entirely if you
                # switch to z-score scaling.
                deconv_layers += [nn.Flatten(), nn.Sigmoid()]

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (self._decoder_channels, self._decoder_length)),
            *deconv_layers,
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
        """Controlled capacity increase (Burgess et al., 2018):
        reconstruction + gamma * |KL - C|, instead of reconstruction + KL.
        `kl_loss` reported here is still the plain, unweighted KL -- gamma/C
        only shape how it enters `loss`, the actual optimized objective."""
        recon_loss = reconstruction_loss(outputs["recon"], x)
        kl_loss = kl_divergence(outputs["mu"], outputs["logvar"])
        return {
            "loss": recon_loss + self.gamma * torch.abs(kl_loss - self.C),
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
