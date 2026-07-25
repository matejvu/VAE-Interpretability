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
composes the term primitives in training/losses.py into the standard,
unweighted ELBO. That's the only composition currently needed (only
VanillaVAE is implemented); a future variant needing a different
composition (beta-weighted KL, a classification term, ...) means
extending `_compute_loss`, not adding a method back onto the model.

Wiring this up to a specific config/dataset/experiment tracker is the job
of the caller (see scripts/train.py), via plain constructor args and the
optional `on_epoch_end` callback.
"""

import time
from pathlib import Path

import torch

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
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = checkpoint_every
        self.early_stopping_patience = early_stopping_patience
        self.input_key = input_key

        self.best_val_loss = float("inf")

    def _extract_input(self, batch):
        x = batch[self.input_key] if isinstance(batch, dict) else batch
        return x.to(self.device)

    def _compute_loss(self, x, outputs):
        """Standard, unweighted ELBO: reconstruction + KL to N(0, I)."""
        recon_loss = reconstruction_loss(outputs["recon"], x)
        kl_loss = kl_divergence(outputs["mu"], outputs["logvar"])
        return {
            "loss": recon_loss + kl_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def _run_epoch(self, loader, train_mode):
        self.model.train(train_mode)

        totals = {}
        n_batches = 0
        with torch.set_grad_enabled(train_mode):
            for batch in loader:
                x = self._extract_input(batch)
                outputs = self.model(x)
                losses = self._compute_loss(x, outputs)

                if train_mode:
                    self.optimizer.zero_grad()
                    losses["loss"].backward()
                    self.optimizer.step()

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
        on_epoch_end(epoch, train_metrics, val_metrics, epoch_time)
        after each epoch -- e.g. for experiment-tracker logging or
        printing, which this class deliberately doesn't do itself.
        `epoch_time` is wall-clock seconds for that epoch's train+val
        passes (excludes checkpoint I/O, which happens after the callback).
        """
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            start_time = time.perf_counter()
            train_metrics = self._run_epoch(train_loader, train_mode=True)
            val_metrics = self._run_epoch(val_loader, train_mode=False)
            epoch_time = time.perf_counter() - start_time

            if on_epoch_end is not None:
                on_epoch_end(epoch, train_metrics, val_metrics, epoch_time)

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
        """Run one no-grad pass over `loader`. Loads checkpoint_dir/best.pt
        first unless load_best=False (e.g. to evaluate the in-memory model
        as-is)."""
        if load_best:
            self.load_checkpoint(self.checkpoint_dir / "best.pt")
        return self._run_epoch(loader, train_mode=False)
