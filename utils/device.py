"""utils/device.py — CPU/GPU device selection utility."""

import torch


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> str:
    """Human-readable device description."""
    device = get_device()
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        return f"GPU — {name} ({mem} MB)"
    if device.type == "mps":
        return "GPU — Apple Silicon (MPS)"
    return "CPU"
