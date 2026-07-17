"""Single-GPU device abstraction. One place to change if setup ever changes."""
import torch


def get_device(config: dict | None = None) -> torch.device:
    requested = None
    if config is not None:
        requested = config.get("training", {}).get("device")

    if requested == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
