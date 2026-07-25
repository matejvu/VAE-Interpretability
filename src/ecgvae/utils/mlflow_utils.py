"""MLflow setup helpers — tracking URI, experiment selection, git commit tag."""
import os
import subprocess
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient

from ecgvae.utils.config import flatten


def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def setup_mlflow(config: dict) -> None:
    """Point mlflow at the tracking server (e.g. DagsHub) via env var, and
    select/create the experiment named in the config."""
    tracking_uri_env_var = config["mlflow"].get("tracking_uri_env_var", "MLFLOW_TRACKING_URI")
    tracking_uri = os.environ.get(tracking_uri_env_var)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        print(
            f"Warning: {tracking_uri_env_var} not set in environment — "
            "falling back to local ./mlruns"
        )
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


def log_run_metadata(config: dict, data_version: str | None = None) -> None:
    """Call once at the start of a run: logs full config as params + useful tags."""
    mlflow.log_params(flatten(config))
    mlflow.set_tag("git_commit", get_git_commit_hash())
    mlflow.set_tag("variant", config["model"]["type"])
    if data_version:
        mlflow.set_tag("data_version", data_version)
    elif "data_dir" in config.get("data", {}):
        # fall back to inferring from the folder name, e.g. .../v2_wavelet_seg360
        mlflow.set_tag("data_version", os.path.basename(config["data"]["data_dir"]))


def register_trained_model(model, model_type: str, config: dict) -> None:
    """Register `model` (should already hold the checkpoint to keep, e.g.
    Trainer's best-val weights) as a new version under `model_type` in the
    Model Registry. Tags describe the model itself (architecture/size),
    not run performance -- that's already on the linked run."""
    # pickle, not the "pt2" (torch.export) default -- pt2 needs a traced
    # single-tensor signature, and forward() here returns a dict.
    model_info = mlflow.pytorch.log_model(
        model, name="model", registered_model_name=model_type, serialization_format="pickle"
    )
    version = model_info.registered_model_version

    # Full layer-by-layer structure (nn.Module's own __repr__) -- too long
    # for the version description below, so it goes alongside the model
    # files as an artifact instead.
    mlflow.log_text(str(model), "model/architecture.txt")

    client = MlflowClient()
    model_cfg = config["model"]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # hidden_dims omitted: it's VanillaVAE-specific, not part of the
    # BaseVAE contract every variant shares (input_length/latent_dim are).
    client.update_model_version(
        name=model_type,
        version=version,
        description=(
            f"{model_type} -- input_length={model_cfg.get('input_length')}, "
            f"latent_dim={model_cfg.get('latent_dim')}, total_params={total_params:,}"
        ),
    )

    tags = {
        "input_length": model_cfg.get("input_length"),
        "latent_dim": model_cfg.get("latent_dim"),
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    for key, value in tags.items():
        client.set_model_version_tag(name=model_type, version=version, key=key, value=str(value))
