"""
fb_vanilla_vae.py

Standard (unconditional) VAE: single Gaussian encoder/decoder pair, with
the original VAE paper's loss (Kingma & Welling, 2013) -- reconstruction +
KL to N(0, I). Supports an optional per-dimension free-bits floor
(Kingma et al., 2016) via `lambda_bits`; leave it unset (None -> 0.0) for
the plain, unfloored ELBO.
See BaseVAE (base_vae.py) for the shared reparameterize/forward/sample/
reconstruct logic and the encode/decode/loss_function contract.

Encoder: 4x [Conv1d -> BatchNorm1d -> ReLU], stride 2 each (256 -> 16
timesteps for the default input_length), no pooling -- downsampling is
learned via strided convolution rather than a fixed max-pool operator.
Decoder mirrors this with ConvTranspose1d.
"""

import torch
import torch.nn as nn

from ecgvae.models.base_vae import BaseVAE
from ecgvae.training.losses import free_bits_kl_divergence, reconstruction_loss


class FBVanillaVAE(BaseVAE):
    def __init__(
        self,
        input_length: int,
        latent_dim: int,
        lambda_bits: float | None = None,
        base_channels: int = 16,
        n_conv_layers: int = 4,
    ):
        super().__init__(input_length, latent_dim)

        # None -> 0.0: a zero floor is a no-op on KL (KL_i >= 0 always),
        # so this recovers the plain unweighted ELBO by default.
        self.lambda_bits = lambda_bits if lambda_bits is not None else 0.0

        # --- channel schedule: 1 -> 16 -> 32 -> 64 -> 64, stride 2 each ---
        channels = [1] + [base_channels * (2 ** min(i, 2)) for i in range(n_conv_layers)]
        # e.g. base_channels=16, n_conv_layers=4 -> [1, 16, 32, 64, 64]

        conv_layers = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
        conv_stack = nn.Sequential(*conv_layers)

        # Infer flattened bottleneck size generically (robust to input_length
        # or n_conv_layers changes) rather than hardcoding 64*16=1024.
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

        rev_channels = channels[::-1]  # [64, 64, 32, 16, 1]
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
        """
        VAE ELBO: reconstruction + (optionally floored) KL to N(0, I).

        kl_weight (optional, via kwargs): external annealing multiplier
        applied to the KL term, e.g. from a linear/sigmoid ramp in your
        training loop. Defaults to 1.0 (no annealing) if not supplied --
        this class stays agnostic to *how* you schedule it.
        """
        recon_loss = reconstruction_loss(outputs["recon"], x)
        kl_loss = free_bits_kl_divergence(
            outputs["mu"], outputs["logvar"], lambda_bits=self.lambda_bits
        )
        kl_weight = kwargs.get("kl_weight", 1.0)
        return {
            "loss": recon_loss + kl_weight * kl_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }