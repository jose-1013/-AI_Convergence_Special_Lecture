from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping, Sequence
from typing import Any


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization helpers. Install it with "
            "'uv add matplotlib' or 'uv run --with matplotlib ...'."
        ) from exc
    return plt


def _resolve_title(sample: Mapping[str, Any]) -> str:
    for key in ("filename", "cocoid", "id", "image_url", "url"):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return "sample"


def _resolve_identifier(sample: Mapping[str, Any]) -> str:
    for key in ("image_url", "url", "cocoid", "id", "filename"):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return "sample"


def _format_metadata(sample: Mapping[str, Any]) -> str | None:
    metadata_parts: list[str] = []
    for key in ("cocoid", "id"):
        value = sample.get(key)
        if value is not None:
            metadata_parts.append(f"{key}: {value}")
    for key in ("image_url", "url"):
        value = sample.get(key)
        if value is not None:
            metadata_parts.append(f"{key}: {value}")
            break
    return " | ".join(metadata_parts) if metadata_parts else None


def _format_caption_block(
    caption: str,
    *,
    predicted_caption: str | None,
    caption_label: str,
    metadata: str | None,
    width: int = 42,
) -> str:
    lines = [f"{caption_label}: {textwrap.fill(caption, width=width)}"]
    if predicted_caption is not None:
        lines.append(f"PRED: {textwrap.fill(predicted_caption, width=width)}")
    if metadata:
        lines.append(textwrap.fill(metadata, width=width))
    return "\n".join(lines)


def _draw_caption_card(
    ax: Any,
    *,
    sample: Mapping[str, Any],
    predicted_caption: str | None,
    caption_label: str,
    title: str | None = None,
) -> None:
    ax.imshow(sample["image"])
    ax.axis("off")
    ax.set_title(_resolve_title(sample) if title is None else title)
    ax.text(
        0.02,
        0.02,
        _format_caption_block(
            str(sample["caption"]).strip(),
            predicted_caption=predicted_caption,
            caption_label=caption_label,
            metadata=_format_metadata(sample),
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="white",
        bbox={
            "facecolor": "black",
            "alpha": 0.75,
            "edgecolor": "none",
            "boxstyle": "round,pad=0.4",
        },
        wrap=True,
    )


def _show_image_sequence(sample: Mapping[str, Any]) -> None:
    plt = require_matplotlib()
    figure = plt.figure(figsize=(6.0, 4.8))
    try:
        plt.imshow(sample["image"])
        plt.axis("off")
        plt.show()
    finally:
        plt.close(figure)


def _print_caption_sequence(
    samples: Sequence[Mapping[str, Any]],
    *,
    predicted_captions: Sequence[str | None],
) -> None:
    for sample, predicted_caption in zip(samples, predicted_captions, strict=True):
        print("---")
        print(f"image url(id): {_resolve_identifier(sample)}")
        _show_image_sequence(sample)
        caption = str(sample["caption"]).strip() if predicted_caption is None else str(predicted_caption).strip()
        print(f"caption: {caption}")


def _print_caption_comparison_sequence(
    samples: Sequence[Mapping[str, Any]],
    *,
    prediction_sets: Mapping[str, Sequence[str]],
) -> None:
    labels = list(prediction_sets.keys())
    for row_index, sample in enumerate(samples):
        print("---")
        print(f"image url(id): {_resolve_identifier(sample)}")
        _show_image_sequence(sample)
        for label in labels:
            print(f"{label}: {str(prediction_sets[label][row_index]).strip()}")


def show_caption_cards(
    samples: Sequence[Mapping[str, Any]],
    *,
    predicted_captions: Sequence[str | None] | None = None,
    caption_label: str = "GT",
    cols: int = 2,
    title: str | None = None,
    show: bool = True,
) -> tuple[Any, Any] | None:
    if not samples:
        raise ValueError("samples must contain at least one item.")
    if cols <= 0:
        raise ValueError("cols must be a positive integer.")

    if predicted_captions is None:
        predicted_captions = [None] * len(samples)
    if len(predicted_captions) != len(samples):
        raise ValueError("predicted_captions must match samples length.")

    if show:
        _print_caption_sequence(
            samples,
            predicted_captions=[
                None if predicted_caption is None else str(predicted_caption)
                for predicted_caption in predicted_captions
            ],
        )
        return None

    plt = require_matplotlib()
    rows = math.ceil(len(samples) / cols)
    figure, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 6.2 * rows), squeeze=False)
    flat_axes = axes.ravel()

    for index, (sample, predicted_caption) in enumerate(zip(samples, predicted_captions, strict=True)):
        _draw_caption_card(
            flat_axes[index],
            sample=sample,
            predicted_caption=None if predicted_caption is None else str(predicted_caption),
            caption_label=caption_label,
        )

    for ax in flat_axes[len(samples):]:
        ax.axis("off")

    if title is not None:
        figure.suptitle(title)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    else:
        figure.tight_layout()
    return figure, axes


def show_caption_comparison_cards(
    samples: Sequence[Mapping[str, Any]],
    *,
    prediction_sets: Mapping[str, Sequence[str]],
    caption_label: str = "GT",
    max_items: int = 4,
    show: bool = True,
) -> tuple[Any, Any] | None:
    if not samples:
        raise ValueError("samples must contain at least one item.")
    if not prediction_sets:
        raise ValueError("prediction_sets must contain at least one label.")

    labels = list(prediction_sets.keys())
    limited_samples = list(samples[:max_items])
    for label, predictions in prediction_sets.items():
        if len(predictions) < len(limited_samples):
            raise ValueError(f"prediction set '{label}' must cover all displayed samples.")

    if show:
        _print_caption_comparison_sequence(
            limited_samples,
            prediction_sets=prediction_sets,
        )
        return None

    plt = require_matplotlib()
    figure, axes = plt.subplots(
        len(limited_samples),
        len(labels),
        figsize=(6.2 * len(labels), 6.4 * len(limited_samples)),
        squeeze=False,
    )

    for row_index, sample in enumerate(limited_samples):
        for col_index, label in enumerate(labels):
            _draw_caption_card(
                axes[row_index][col_index],
                sample=sample,
                predicted_caption=str(prediction_sets[label][row_index]),
                caption_label=caption_label,
                title=f"{label} | {_resolve_title(sample)}",
            )

    figure.tight_layout()
    return figure, axes


def show_caption_metric_comparison(
    metrics_by_label: Mapping[str, Mapping[str, float]],
    *,
    title: str | None = None,
    show_table: bool = True,
    show_chart: bool = True,
    metric_order: Sequence[str] = ("bleu", "meteor", "cider_d"),
    show: bool = True,
) -> Any:
    if not metrics_by_label:
        raise ValueError("metrics_by_label must contain at least one item.")

    metric_names = [name for name in metric_order if any(name in metrics for metrics in metrics_by_label.values())]
    if not metric_names:
        raise ValueError("No supported metrics were found.")

    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table
    except ImportError as exc:
        raise ImportError(
            "rich is required for metric table output. Install it with "
            "'uv add rich' or 'uv run --with rich ...'."
        ) from exc

    labels = list(metrics_by_label.keys())
    label_width = max(len("model"), *(len(label) for label in labels))
    metric_width = max(8, *(len(metric_name) for metric_name in metric_names))
    table = Table(
        title=title,
        box=box.SIMPLE,
        show_header=True,
        header_style="bold",
        expand=False,
    )
    table.add_column("model", width=label_width, no_wrap=True)
    for metric_name in metric_names:
        table.add_column(metric_name, justify="right", width=metric_width, no_wrap=True)

    for label in labels:
        table.add_row(
            label,
            *[f"{float(metrics_by_label[label].get(metric_name, 0.0)):.4f}" for metric_name in metric_names],
        )

    if show and show_table:
        Console().print(table)
    return table


def show_lora_cost_comparison(
    full_finetuning: Mapping[str, float | int],
    lora: Mapping[str, float | int],
    *,
    show: bool = True,
) -> tuple[Any, Any]:
    plt = require_matplotlib()
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    labels = ["Full FT", "LoRA"]
    values = [float(full_finetuning["total_bytes"]), float(lora["total_bytes"])]
    bars = axis.bar(labels, values, color=["#d95f02", "#1b9e77"])
    axis.set_ylabel("Estimated bytes")
    axis.set_title("Full Fine-tuning vs LoRA Memory Cost")
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    if show:
        plt.show()
    return figure, axis


def show_lora_rank_sweep(
    rank_records: Sequence[Mapping[str, float | int]],
    *,
    show: bool = True,
) -> tuple[Any, Any]:
    if not rank_records:
        raise ValueError("rank_records must contain at least one item.")

    plt = require_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    ranks = [int(record["rank"]) for record in rank_records]
    trainable_ratios = [float(record["trainable_ratio"]) for record in rank_records]
    approximation_errors = [float(record["approximation_error"]) for record in rank_records]

    axes[0].plot(ranks, trainable_ratios, marker="o")
    axes[0].set_title("Rank vs Trainable Ratio")
    axes[0].set_xlabel("rank r")
    axes[0].set_ylabel("trainable ratio")

    axes[1].plot(ranks, approximation_errors, marker="o")
    axes[1].set_title("Rank vs Approximation Error")
    axes[1].set_xlabel("rank r")
    axes[1].set_ylabel("Frobenius error")

    figure.tight_layout()
    if show:
        plt.show()
    return figure, axes


def show_lora_alpha_sweep(
    alpha_records: Sequence[Mapping[str, float | int]],
    *,
    show: bool = True,
) -> tuple[Any, Any]:
    if not alpha_records:
        raise ValueError("alpha_records must contain at least one item.")

    plt = require_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    alphas = [int(record["alpha"]) for record in alpha_records]
    update_norms = [float(record["update_norm"]) for record in alpha_records]
    output_delta_norms = [float(record["output_delta_norm"]) for record in alpha_records]

    axes[0].plot(alphas, update_norms, marker="o")
    axes[0].set_title("Alpha vs Update Norm")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("update norm")

    axes[1].plot(alphas, output_delta_norms, marker="o")
    axes[1].set_title("Alpha vs Output Delta Norm")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("output delta norm")

    figure.tight_layout()
    if show:
        plt.show()
    return figure, axes


def show_lora_analysis_dashboard(
    cost_summary: Mapping[str, Mapping[str, float | int]],
    rank_records: Sequence[Mapping[str, float | int]],
    alpha_records: Sequence[Mapping[str, float | int]],
    *,
    show: bool = True,
) -> tuple[Any, Any]:
    plt = require_matplotlib()
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.5))

    full_bytes = float(cost_summary["full_finetuning"]["total_bytes"])
    lora_bytes = float(cost_summary["lora"]["total_bytes"])
    axes[0].bar(["Full FT", "LoRA"], [full_bytes, lora_bytes], color=["#d95f02", "#1b9e77"])
    axes[0].set_title("Estimated memory cost")
    axes[0].set_ylabel("bytes")

    ranks = [int(record["rank"]) for record in rank_records]
    ratios = [float(record["trainable_ratio"]) for record in rank_records]
    errors = [float(record["approximation_error"]) for record in rank_records]
    axes[1].plot(ranks, ratios, marker="o", label="trainable ratio")
    axes[1].plot(ranks, errors, marker="s", label="approximation error")
    axes[1].set_title("Rank sweep")
    axes[1].legend()

    alphas = [int(record["alpha"]) for record in alpha_records]
    norms = [float(record["update_norm"]) for record in alpha_records]
    delta_norms = [float(record["output_delta_norm"]) for record in alpha_records]
    axes[2].plot(alphas, norms, marker="o", label="update norm")
    axes[2].plot(alphas, delta_norms, marker="s", label="output delta norm")
    axes[2].set_title("Alpha sweep")
    axes[2].legend()

    figure.tight_layout()
    if show:
        plt.show()
    return figure, list(axes)


def show_lora_rank_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    zero_shot_metrics: Mapping[str, float] | None = None,
    metric_order: Sequence[str] = ("bleu", "meteor", "cider_d"),
    show: bool = True,
) -> tuple[Any, Any]:
    if not rows:
        raise ValueError("rows must contain at least one item.")

    plt = require_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    flat_axes = axes.ravel()
    ranks = [str(row["rank"]) for row in rows]
    training_times = [
        float(row["training_time_sec"] if "training_time_sec" in row else row["time_sec"])
        for row in rows
    ]

    flat_axes[0].bar(ranks, [float(row["trainable_ratio"]) for row in rows], color="#1b9e77")
    flat_axes[0].set_title("Trainable ratio")

    flat_axes[1].bar(ranks, training_times, color="#7570b3")
    flat_axes[1].set_title("Training time (sec)")

    flat_axes[2].bar(ranks, [float(row["eval_loss"]) for row in rows], color="#d95f02")
    flat_axes[2].set_title("Eval loss")

    metric_axis = flat_axes[3]
    metric_axis.set_title("Caption metrics by rank")
    for metric_name in metric_order:
        values: list[float] = []
        for row in rows:
            caption_scores = row.get("caption_scores", {})
            if not isinstance(caption_scores, Mapping):
                continue
            fine_tuned_scores = caption_scores.get("fine_tuned")
            if isinstance(fine_tuned_scores, Mapping) and metric_name in fine_tuned_scores:
                values.append(float(fine_tuned_scores[metric_name]))
        if len(values) == len(rows):
            metric_axis.plot(ranks, values, marker="o", label=f"{metric_name} (fine-tuned)")
        if zero_shot_metrics and metric_name in zero_shot_metrics:
            metric_axis.axhline(
                float(zero_shot_metrics[metric_name]),
                linestyle="--",
                linewidth=1.2,
                label=f"{metric_name} zero-shot",
            )
    metric_axis.legend()

    figure.tight_layout()
    if show:
        plt.show()
    return figure, list(flat_axes)


__all__ = [
    "require_matplotlib",
    "show_caption_cards",
    "show_caption_comparison_cards",
    "show_caption_metric_comparison",
    "show_lora_alpha_sweep",
    "show_lora_analysis_dashboard",
    "show_lora_cost_comparison",
    "show_lora_rank_summary",
    "show_lora_rank_sweep",
]
