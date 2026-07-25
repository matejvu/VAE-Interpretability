"""
train.py

Wiring only: reads a YAML config (see configs/*.yaml), builds the model
variant named in config["model"]["type"], builds the MIT-BIH dataloaders
(via ecgvae.data.mitbih.build_dataloaders), and hands them to the
dataset/config-agnostic Trainer (ecgvae.training.trainer.Trainer) which
actually runs the training loop. This file owns everything adjustable --
which config, which dataloaders, mlflow -- Trainer owns none of it.

Usage:
    python scripts/train.py --config configs/vanilla_vae.yaml

Only "vanilla_vae" is runnable today -- BetaVAE and SemiSupervisedVAE
(configs/beta_vae.yaml, configs/semi_supervised.yaml) are still empty
stub files in src/ecgvae/models/, so selecting those configs raises
NotImplementedError instead of silently doing the wrong thing.
"""

import argparse
from pathlib import Path

import mlflow
import torch
from dotenv import load_dotenv

from ecgvae.data.mitbih import build_dataloaders
from ecgvae.models.vanilla_vae import VanillaVAE
from ecgvae.training.trainer import Trainer
from ecgvae.utils.config import load_config
from ecgvae.utils.device import get_device
from ecgvae.utils.mlflow_utils import log_run_metadata, register_trained_model, setup_mlflow
from ecgvae.utils.seeding import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Load PROJECT_ROOT/.env (e.g. MLFLOW_TRACKING_URI) explicitly by absolute
# path -- not just load_dotenv()'s cwd-relative default -- so it's found
# regardless of which directory this script is launched from.
load_dotenv(PROJECT_ROOT / ".env")

# Only variants with an actual implementation in src/ecgvae/models/ are
# listed here. beta_vae.py and semi_supervised.py are currently empty
# stub files, so their configs aren't runnable yet.
MODEL_REGISTRY = {
    "vanilla_vae": VanillaVAE,
}

OPTIMIZER_REGISTRY = {
    "adam": torch.optim.Adam,
}


def build_model(model_cfg: dict) -> torch.nn.Module:
    model_type = model_cfg["type"]
    if model_type not in MODEL_REGISTRY:
        raise NotImplementedError(
            f"model.type='{model_type}' has no implementation yet in "
            f"src/ecgvae/models/ (only {list(MODEL_REGISTRY)} are runnable)."
        )
    model_cls = MODEL_REGISTRY[model_type]

    # input_length/latent_dim are part of the BaseVAE contract, so every
    # variant's constructor takes them -- named explicitly here. Everything
    # else (hidden_dims, beta, num_classes, ...) is variant-specific, so it's
    # passed through as-is; a real mismatch with the constructor's signature
    # still fails loudly with Python's own TypeError.
    extra_kwargs = {
        k: v for k, v in model_cfg.items() if k not in ("type", "input_length", "latent_dim")
    }
    return model_cls(
        input_length=model_cfg["input_length"],
        latent_dim=model_cfg["latent_dim"],
        **extra_kwargs,
    )


def build_optimizer(training_cfg: dict, params) -> torch.optim.Optimizer:
    optimizer_name = training_cfg["optimizer"]
    if optimizer_name not in OPTIMIZER_REGISTRY:
        raise NotImplementedError(
            f"training.optimizer='{optimizer_name}' is not supported "
            f"(only {list(OPTIMIZER_REGISTRY)} are)."
        )
    optimizer_cls = OPTIMIZER_REGISTRY[optimizer_name]
    return optimizer_cls(
        params, lr=training_cfg["lr"], weight_decay=training_cfg["weight_decay"]
    )


def make_epoch_logger():
    """mlflow + stdout logging callback for Trainer.fit(on_epoch_end=...).
    Kept here (not in Trainer) since Trainer knows nothing about mlflow."""

    def on_epoch_end(epoch, train_metrics, val_metrics, epoch_time):
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"(recon={train_metrics['recon_loss']:.4f}, kl={train_metrics['kl_loss']:.4f})  "
            f"val_loss={val_metrics['loss']:.4f} "
            f"(recon={val_metrics['recon_loss']:.4f}, kl={val_metrics['kl_loss']:.4f})  "
            f"time={epoch_time:.2f}s"
        )
        # train_metrics/val_metrics already carry recon_loss/kl_loss alongside
        # loss (see Trainer._compute_loss), so this logs all three per split --
        # train_recon_loss, train_kl_loss, val_recon_loss, val_kl_loss, etc.
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()}, step=epoch)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)
        mlflow.log_metric("epoch_time_sec", epoch_time, step=epoch)

    return on_epoch_end


def train(config: dict) -> None:
    set_seed(config["training"]["seed"])
    device = get_device(config)
    print(f"Using device: {device}")

    loaders, _ = build_dataloaders(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        normalize=True,
    )

    model = build_model(config["model"]).to(device)
    optimizer = build_optimizer(config["training"], model.parameters())

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR / config["mlflow"]["experiment_name"],
        checkpoint_every=config["training"]["checkpoint_every"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
    )

    setup_mlflow(config)
    with mlflow.start_run():
        log_run_metadata(config)

        trainer.fit(
            loaders["train"],
            loaders["val"],
            epochs=config["training"]["epochs"],
            on_epoch_end=make_epoch_logger(),
        )

        test_metrics = trainer.evaluate(loaders["test"], load_best=True)
        print(f"[test, best checkpoint] loss={test_metrics['loss']:.4f}")
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # `model` now holds the best-val checkpoint's weights (evaluate()
        # just loaded them) -- register exactly that, under model_type so
        # different hyperparameters land as new versions of the same
        # registered model rather than separate ones.
        register_trained_model(model, config["model"]["type"], config)


def main():
    parser = argparse.ArgumentParser(description="Train an ecgvae model variant from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a config YAML (e.g. configs/vanilla_vae.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
