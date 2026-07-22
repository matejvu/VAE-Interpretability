"""
mitbih.py

Dataset + DataLoader utilities for the processed MIT-BIH beat data.
Import from this module wherever you need train/val/test data.

Usage:
    from mitbih import build_dataloaders

    loaders, datasets = build_dataloaders(batch_size=64, normalize=True)
    for batch in loaders["train"]:
        x = batch["waveform"]          # (B, WINDOW_LEN), scaled to [0, 1]
        y = batch["aami_label"]        # (B,) int labels

    # Later, to view a reconstruction in real amplitude units:
    real_x = datasets["train"].denormalize(x[0].numpy(), batch["record_id"][0])

Normalization design:
  - Stats are NOT precomputed anywhere upstream (extraction script saves
    raw, detrended-but-unscaled windows only).
  - Per-record min-max stats are computed ONCE, in ECGBeatDataset.__init__,
    only from the records present in that split -- so a train Dataset's
    stats come purely from train patients, a val Dataset's purely from val
    patients, etc. No cross-split leakage possible by construction.
  - If normalize=False, no stats are computed at all -- zero added cost.
  - Scaling is per-record min-max using robust percentiles (not literal
    min/max) to avoid a single noisy beat setting the whole record's scale.
  - Reversibility is provided via Dataset.denormalize(x, record_id), so
    reconstructions / latent traversals can be shown in real (detrended,
    mV-ish) amplitude units rather than [0, 1] -- important for the
    interpretability analysis, where normalized units would be much less
    legible in figures.
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# AAMI class -> integer label, fixed ordering used consistently across the
# project (model outputs, confusion matrices, etc. should all use this).
AAMI_LABEL_TO_IDX = {"N": 0, "S": 1, "V": 2, "F": 3}
IDX_TO_AAMI_LABEL = {v: k for k, v in AAMI_LABEL_TO_IDX.items()}


class ECGBeatDataset(Dataset):
    def __init__(self, split, windows_path=None, meta_path=None, splits_path=None,
                 normalize=True, lower_pct=1, upper_pct=99, verbose=True):
        """
        Args:
            split: one of "train", "val", "test".
            normalize: if True, compute per-record min-max stats (once,
                here, from this split's own records only) and apply
                (x - lo) / (hi - lo) per sample. If False, no stats are
                computed at all -- samples returned in raw units.
            lower_pct, upper_pct: percentiles used for robust min-max.
                Set to 0/100 for literal min-max.
        """
        assert split in ("train", "val", "test")
        self.split = split
        self.normalize = normalize
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

        windows_path = windows_path or PROCESSED_DIR / "beat_windows.npy"
        meta_path = meta_path or PROCESSED_DIR / "beat_metadata.csv"
        splits_path = splits_path or PROCESSED_DIR / "patient_splits.json"

        all_windows = np.load(windows_path)
        meta_df = pd.read_csv(meta_path, dtype={"record_id": str, "patient_id": str})

        with open(splits_path) as f:
            patient_splits = json.load(f)

        meta_df["split"] = meta_df["patient_id"].map(patient_splits)
        split_df = meta_df[meta_df["split"] == split].reset_index(drop=True)

        if len(split_df) == 0:
            raise ValueError(f"No beats found for split='{split}' -- check "
                              f"patient_splits.json and beat_metadata.csv agree.")

        self.window_idx = split_df["window_idx"].to_numpy()
        self.windows = all_windows[self.window_idx]  # (n_split, WINDOW_LEN)
        self.record_ids = split_df["record_id"].to_numpy()
        self.patient_ids = split_df["patient_id"].to_numpy()
        self.aami_labels = split_df["aami_label"].map(AAMI_LABEL_TO_IDX).to_numpy()

        self.norm_stats = None  # record_id -> (lo, hi)
        if self.normalize:
            self.norm_stats = self._compute_norm_stats(verbose=verbose)

        if verbose:
            print(f"[{split}] {len(self.windows)} beats, "
                  f"{len(set(self.record_ids))} records, "
                  f"{len(set(self.patient_ids))} patients, "
                  f"normalize={self.normalize}")

    def _compute_norm_stats(self, verbose=True):
        """Per-record {lo, hi} percentile stats, computed once from this
        split's own beats only. Cheap -- single pass, no disk caching needed."""
        stats = {}
        for record_id in np.unique(self.record_ids):
            mask = self.record_ids == record_id
            values = self.windows[mask].ravel()
            lo = float(np.percentile(values, self.lower_pct))
            hi = float(np.percentile(values, self.upper_pct))
            stats[record_id] = (lo, hi)
        if verbose:
            print(f"[{self.split}] computed normalization stats for "
                  f"{len(stats)} records (percentiles {self.lower_pct}/{self.upper_pct})")
        return stats

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx].astype(np.float32)
        record_id = self.record_ids[idx]

        if self.normalize:
            x = self._scale(x, record_id)

        return {
            "waveform": torch.from_numpy(x).float(),   # (WINDOW_LEN,)
            "aami_label": int(self.aami_labels[idx]),
            "record_id": record_id,
            "patient_id": self.patient_ids[idx],
        }

    # -- reversible scaling utilities -----------------------------------

    def _scale(self, x, record_id):
        """Apply (x - lo) / (hi - lo) using this record's cached stats."""
        lo, hi = self.norm_stats[record_id]
        denom = (hi - lo) if (hi - lo) > 1e-8 else 1.0
        x = (x - lo) / denom
        # Percentile bounds can be exceeded by the true tail extremes --
        # clip to keep values in [0, 1] for a bounded (sigmoid) decoder.
        return np.clip(x, 0.0, 1.0)

    def denormalize(self, x, record_id):
        """
        Invert the scaling transform for a sample (numpy array or torch
        tensor), given its record_id. Use this to view reconstructions /
        latent traversals in real (detrended, mV-ish) amplitude units.
        No-op if normalize=False.
        """
        if not self.normalize:
            return x
        lo, hi = self.norm_stats[record_id]
        denom = (hi - lo) if (hi - lo) > 1e-8 else 1.0
        return x * denom + lo


def build_dataloaders(batch_size=64, normalize=True, num_workers=0,
                       lower_pct=1, upper_pct=99, verbose=True):
    """
    Convenience builder: constructs train/val/test ECGBeatDataset instances
    and wraps each in a DataLoader.

    Returns:
        loaders: dict[str, DataLoader]   -- keys "train", "val", "test"
        datasets: dict[str, ECGBeatDataset]  -- keep these around, e.g. for
            datasets["train"].denormalize(x, record_id) when visualizing.
    """
    datasets = {
        split: ECGBeatDataset(
            split=split, normalize=normalize,
            lower_pct=lower_pct, upper_pct=upper_pct, verbose=verbose,
        )
        for split in ("train", "val", "test")
    }

    loaders = {
        split: DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, ds in datasets.items()
    }

    return loaders, datasets


if __name__ == "__main__":
    loaders, datasets = build_dataloaders(batch_size=32)

    for split, loader in loaders.items():
        batch = next(iter(loader))
        print(f"\n[{split}] batch waveform shape: {batch['waveform'].shape}")
        print(f"[{split}] batch labels: {batch['aami_label'][:8]}")

        # Sanity check: denormalize round-trip on one sample.
        x_norm = batch["waveform"][0].numpy()
        rid = batch["record_id"][0]
        x_real = datasets[split].denormalize(x_norm, rid)
        print(f"[{split}] sample record={rid}, "
              f"normalized range=[{x_norm.min():.3f}, {x_norm.max():.3f}], "
              f"denormalized range=[{x_real.min():.3f}, {x_real.max():.3f}]")