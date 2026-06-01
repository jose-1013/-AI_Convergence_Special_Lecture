from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def count_parameters(model: Any, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def build_training_arguments_kwargs(
    *,
    training_arguments_cls: Any,
    output_dir: str,
    config: Any,
    device: Any,
    torch_dtype: Any | None,
) -> dict[str, Any]:
    dtype_name = "" if torch_dtype is None else str(torch_dtype).lower()
    use_bf16 = device.type == "cuda" and "bfloat16" in dtype_name
    use_fp16 = device.type == "cuda" and not use_bf16 and "float16" in dtype_name
    kwargs = {
        "output_dir": output_dir,
        "remove_unused_columns": False,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_strategy": "epoch",
        "report_to": "none",
        "gradient_checkpointing": device.type == "cuda",
        "fp16": use_fp16,
        "bf16": use_bf16,
    }
    seed = getattr(config, "seed", None)
    if seed is not None:
        kwargs["seed"] = seed
        kwargs["data_seed"] = seed
    parameter_names = inspect.signature(training_arguments_cls.__init__).parameters
    if "disable_tqdm" in parameter_names:
        kwargs["disable_tqdm"] = False
    if "eval_strategy" in parameter_names:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return kwargs


__all__ = [
    "build_training_arguments_kwargs",
    "count_parameters",
    "save_json",
]
