"""
beta_vae.py

Same conv/deconv architecture as FBVanillaVAE/FixedChannelVAE (see
fb_vanilla_vae.py, fixed_channel_vae.py), with the original beta-VAE loss
(Higgins et al., 2017): reconstruction + beta * KL, instead of plain KL
(beta=1 recovers the vanilla ELBO) or the |KL - C| capacity constraint
FixedChannelVAE uses.

Note beta-VAE is a special case of FixedChannelVAE's loss at C=0: since
KL >= 0 always, |KL - 0| = KL, so gamma * |KL - 0| = gamma * KL -- the same
form as beta * KL, with gamma playing beta's role. The two aren't meant to
be combined in one model; BetaVAE takes only `beta`.

All architecture knobs (base_channels, n_conv_layers, latent_dim) are real
constructor params with defaults, not fixed constants -- override any of
them from your YAML config as needed.

See BaseVAE (base_vae.py) for the shared reparameterize/forward/sample/
reconstruct logic and the encode/decode/loss_function contract.
"""

import torch
import torch.nn as nn

from ecgvae.models.base_vae import BaseVAE
from ecgvae.training.losses import kl_divergence, reconstruction_loss


class BetaVAE(BaseVAE):
    def __init__(
        self,
        input_length: int,
        latent_dim: int,
        beta: float | None = None,
        base_channels: int = 16,
        n_conv_layers: int = 4,
    ):
        super().__init__(input_length, latent_dim)

        if beta is None:
            raise ValueError("BetaVAE requires beta (no default -- set it in your config).")
        self.beta = beta

        # --- channel schedule: 1 -> base_channels -> ... , stride 2 each ---
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
        """beta-VAE ELBO: reconstruction + beta * KL to N(0, I)."""
        recon_loss = reconstruction_loss(outputs["recon"], x)
        kl_loss = kl_divergence(outputs["mu"], outputs["logvar"])
        return {
            "loss": recon_loss + self.beta * kl_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
