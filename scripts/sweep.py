"""
sweep.py

Grid search over hyperparameters, driven by a small sweep-spec YAML (see
configs/sweeps/*.yaml) -- a different schema from configs/*.yaml (which
train.py reads directly), so there's no ambiguity between a config value
that happens to be a list (e.g. model.hidden_dims) and a list of
candidates to search over.

Each combination runs as its own `python scripts/train.py --config <tmp>`
subprocess, not an in-process call to train() -- so one bad combination
(NaN loss, OOM, ...) can't take down the rest of the sweep, and CUDA
memory/state cleanly resets between runs.

Sweep-spec format:
    base_config: configs/vanilla_vae.yaml   # loaded via load_config (extends-merged)
    experiment_name: vanilla-vae-sweep      # optional -- overwrites
                                              # mlflow.experiment_name from
                                              # base_config for every run in
                                              # this sweep, so results land
                                              # in their own experiment,
                                              # separate from regular runs
    grid:
      training.lr: [0.01, 0.001, 0.0001]
      model.latent_dim: [8, 16, 32]

Usage:
    python scripts/sweep.py --sweep-config configs/sweeps/vanilla_vae_sweep.yaml
    python scripts/sweep.py --sweep-config configs/sweeps/vanilla_vae_sweep.yaml --dry-run
"""

import argparse
import copy
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from ecgvae.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"


def set_by_path(config: dict, dotted_key: str, value) -> None:
    """Set config["a"]["b"] = value for dotted_key="a.b"."""
    *parents, leaf = dotted_key.split(".")
    node = config
    for key in parents:
        node = node[key]
    node[leaf] = value


def build_combinations(grid: dict) -> list[dict]:
    """Cartesian product of grid's value-lists, each combo as {dotted_key: value}."""
    keys = list(grid.keys())
    value_lists = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def main():
    parser = argparse.ArgumentParser(description="Grid search over hyperparameters.")
    parser.add_argument("--sweep-config", required=True, help="Path to a sweep-spec YAML")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the combinations without running training"
    )
    args = parser.parse_args()

    with open(args.sweep_config) as f:
        sweep_spec = yaml.safe_load(f)

    base_config = load_config(sweep_spec["base_config"])
    experiment_name = sweep_spec.get("experiment_name")
    combinations = build_combinations(sweep_spec["grid"])

    print(f"{len(combinations)} combination(s) from {args.sweep_config}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, combo in enumerate(combinations, start=1):
            config = copy.deepcopy(base_config)
            for dotted_key, value in combo.items():
                set_by_path(config, dotted_key, value)
            if experiment_name:
                config["mlflow"]["experiment_name"] = experiment_name

            print(f"\n[{i}/{len(combinations)}] {combo}")
            if args.dry_run:
                continue

            tmp_path = Path(tmp_dir) / f"combo_{i:03d}.yaml"
            with open(tmp_path, "w") as f:
                yaml.safe_dump(config, f)

            result = subprocess.run([sys.executable, str(TRAIN_SCRIPT), "--config", str(tmp_path)])
            if result.returncode != 0:
                print(f"  -> FAILED (exit {result.returncode}), continuing with next combination")


if __name__ == "__main__":
    main()
