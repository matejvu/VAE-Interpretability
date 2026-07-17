"""MLflow setup helpers — tracking URI, experiment selection, git commit tag."""
import os
import subprocess
import mlflow

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
