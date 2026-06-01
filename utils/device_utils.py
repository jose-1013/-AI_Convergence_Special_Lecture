from __future__ import annotations

import gc
from typing import Any

import torch


def _parse_cuda_index(device: str | int) -> int | None:
    if isinstance(device, int):
        return device
    if device.isdigit():
        return int(device)
    if device.startswith("cuda:"):
        suffix = device.split(":", maxsplit=1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def resolve_device(device: str | int | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device, str) and device in {"cpu", "cuda"}:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the current environment.")
        return torch.device(device)

    cuda_index = _parse_cuda_index(device)
    if cuda_index is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the current environment.")
        if cuda_index < 0 or cuda_index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested GPU index {cuda_index}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )
        return torch.device(f"cuda:{cuda_index}")

    return torch.device(str(device))


def get_device_info(device: str | int | None = None) -> dict[str, Any]:
    resolved = resolve_device(device)
    info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": resolved,
    }
    if resolved.type == "cuda":
        info["device_index"] = resolved.index if resolved.index is not None else torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(info["device_index"])
    return info


def release_cuda_memory(device: Any, torch_module: Any = torch) -> None:
    if getattr(device, "type", None) != "cuda":
        return
    gc.collect()
    empty_cache = getattr(getattr(torch_module, "cuda", None), "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


__all__ = ["get_device_info", "release_cuda_memory", "resolve_device"]
