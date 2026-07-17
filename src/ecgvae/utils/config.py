"""Simple YAML config loader supporting single-level `extends` inheritance.

Usage:
    from ecgvae.utils.config import load_config
    config = load_config("configs/beta_vae.yaml")
"""
from pathlib import Path
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`. `override` wins on conflicts."""
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f)

    if "extends" in config:
        base_path = path.parent / config["extends"]
        base_config = load_config(base_path)
        config = _deep_merge(base_config, config)

    return config


def flatten(config: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested config dict for mlflow.log_params, e.g.
    {"model": {"latent_dim": 16}} -> {"model.latent_dim": 16}
    """
    items = {}
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten(value, new_key, sep))
        else:
            items[new_key] = value
    return items
