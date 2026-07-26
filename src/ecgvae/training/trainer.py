"""
trainer.py

Generic training loop, deliberately indifferent to where its inputs come
from: it knows nothing about YAML configs, mlflow, or the MIT-BIH dataset
specifically. It only requires:
  - a model implementing the BaseVAE contract (forward(x) -> dict with
    "recon", "mu", "logvar"),
  - an optimizer,
  - plain DataLoaders yielding either a tensor or a dict containing one
    (see `input_key`).

Loss computation is this class's job, not the model's: `_compute_loss`
composes the term primitives in training/losses.py into the standard ELBO,
with an optional linearly-annealed KL weight (see `kl_annealing_epochs`)
to counter posterior collapse. A future variant needing a different
composition (beta-weighted KL, a classification term, ...) means
extending `_compute_loss`, not adding a method back onto the model.

Wiring this up to a specific config/dataset/experiment tracker is the job
of the caller (see scripts/train.py), via plain constructor args and the
optional `on_epoch_end` callback.
"""

import time
from pathlib import Path

import torch

from ecgvae.evaluation.metrics import per_dim_kl, prd
from ecgvae.training.losses import kl_divergence, reconstruction_loss


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        checkpoint_dir,
        checkpoint_every=10,
        early_stopping_patience=None,
        input_key="waveform",
        kl_annealing_epochs=0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = checkpoint_every
        self.early_stopping_patience = early_stopping_patience
        self.input_key = input_key
        # 0/falsy disables annealing (full KL weight from epoch 1, the
        # previous behavior). See _kl_weight_for_epoch.
        self.kl_annealing_epochs = kl_annealing_epochs

        self.best_val_loss = float("inf")

    def _extract_input(self, batch):
        x = batch[self.input_key] if isinstance(batch, dict) else batch
        return x.to(self.device)

    def _kl_weight_for_epoch(self, epoch):
        """Linear KL-annealing schedule: ramps 0 -> 1 over the first
        kl_annealing_epochs epochs, then holds at 1."""
        if not self.kl_annealing_epochs:
            return 1.0
        return min(1.0, epoch / self.kl_annealing_epochs)

    def _compute_loss(self, x, outputs, kl_weight=1.0):
        """ELBO: reconstruction + kl_weight * KL to N(0, I). kl_loss here is
        always the raw, unweighted KL (for diagnostics/comparability) --
        kl_weight only scales its contribution to `loss`, the actual
        optimized objective."""
        recon_loss = reconstruction_loss(outputs["recon"], x)
        kl_loss = kl_divergence(outputs["mu"], outputs["logvar"])
        return {
            "loss": recon_loss + kl_weight * kl_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def _count_active_units(self, val_loader, threshold=0.02):
        """Posterior-collapse diagnostic: how many latent dims have per-dim
        KL above `threshold`, averaged over one batch from val_loader (val
        only -- this is a snapshot of the current model, not a training
        signal, so a train-set version isn't needed)."""
        self.model.eval()
        with torch.no_grad():
            x = self._extract_input(next(iter(val_loader)))
            mu, logvar = self.model.encode(x)
            kl_dims = per_dim_kl(mu, logvar)
            active = (kl_dims > threshold).sum().item()
        self.model.train()
        print(f"\tactive_units: {active}")
        print(f"\tkl_per_dim: {kl_dims}")
        return active

    def _run_epoch(self, loader, train_mode, kl_weight=1.0, compute_prd=False):
        self.model.train(train_mode)

        totals = {}
        n_batches = 0
        with torch.set_grad_enabled(train_mode):
            for batch in loader:
                x = self._extract_input(batch)
                outputs = self.model(x)
                losses = self._compute_loss(x, outputs, kl_weight=kl_weight)

                if train_mode:
                    self.optimizer.zero_grad()
                    losses["loss"].backward()
                    self.optimizer.step()

                if compute_prd:
                    # Reconstruction-quality check (see evaluation/metrics.prd),
                    # not part of the optimized objective -- only computed by
                    # evaluate(), never during fit()'s per-epoch train/val passes.
                    losses["prd"] = prd(outputs["recon"], x)

                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + value.item()
                n_batches += 1

        return {key: total / n_batches for key, total in totals.items()}

    def _save_checkpoint(self, path, epoch, val_loss, include_optimizer):
        payload = {"epoch": epoch, "model_state_dict": self.model.state_dict(), "val_loss": val_loss}
        if include_optimizer:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def fit(self, train_loader, val_loader, epochs, on_epoch_end=None):
        """
        Run up to `epochs` epochs of train/val, tracking the best-val
        checkpoint and (optionally) stopping early. `on_epoch_end`, if
        given, is called as
        on_epoch_end(epoch, train_metrics, val_metrics, epoch_time, kl_weight, active_units)
        after each epoch -- e.g. for experiment-tracker logging or
        printing, which this class deliberately doesn't do itself.
        `epoch_time` is wall-clock seconds for that epoch's train+val
        passes (excludes checkpoint I/O, which happens after the callback).
        `kl_weight` is this epoch's annealing weight (same value used for
        both the train and val pass); train_metrics/val_metrics deliberately
        don't carry it themselves -- it's not a per-split metric.
        `active_units` is the val-only posterior-collapse diagnostic (see
        _count_active_units) -- also not a per-split metric.
        """
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            kl_weight = self._kl_weight_for_epoch(epoch)
            start_time = time.perf_counter()
            train_metrics = self._run_epoch(train_loader, train_mode=True, kl_weight=kl_weight)
            val_metrics = self._run_epoch(val_loader, train_mode=False, kl_weight=kl_weight)
            epoch_time = time.perf_counter() - start_time

            active_units = self._count_active_units(val_loader)

            if on_epoch_end is not None:
                on_epoch_end(epoch, train_metrics, val_metrics, epoch_time, kl_weight, active_units)

            val_loss = val_metrics["loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                epochs_without_improvement = 0
                self._save_checkpoint(
                    self.checkpoint_dir / "best.pt", epoch, val_loss, include_optimizer=False
                )
            else:
                epochs_without_improvement += 1

            if self.checkpoint_every and epoch % self.checkpoint_every == 0:
                self._save_checkpoint(
                    self.checkpoint_dir / f"epoch_{epoch:03d}.pt", epoch, val_loss, include_optimizer=True
                )

            if (
                self.early_stopping_patience is not None
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                break

        return self.best_val_loss

    def evaluate(self, loader, load_best=True):
        """Run one no-grad pass over `loader`, including PRD alongside
        loss/recon_loss/kl_loss. Loads checkpoint_dir/best.pt first unless
        load_best=False (e.g. to evaluate the in-memory model as-is)."""
        if load_best:
            self.load_checkpoint(self.checkpoint_dir / "best.pt")
        return self._run_epoch(loader, train_mode=False, compute_prd=True)
