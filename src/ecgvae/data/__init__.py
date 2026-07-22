"""
Makes data/ an importable subpackage. Re-exports build_dataloaders --
the one thing training/evaluation scripts actually need to import
repeatedly -- so callers can do `from ecgvae.data import build_dataloaders`
instead of reaching into mitbih.py directly.

preprocess_mitbih.py and split_data.py are one-off scripts (run once to
generate processed data / patient_splits.json), not re-imported
repeatedly, so they're deliberately left out of this re-export.
"""

from ecgvae.data.mitbih import build_dataloaders

__all__ = ["build_dataloaders"]