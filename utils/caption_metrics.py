from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from collections.abc import Iterable, Sequence
from importlib import import_module
import io
import math
from typing import Any

SUPPORTED_METRICS = ("bleu", "meteor", "cider_d")


def _normalize_predictions(predictions: Sequence[str]) -> list[str]:
    normalized = [str(prediction).strip() for prediction in predictions]
    if not normalized:
        raise ValueError("predictions must not be empty.")
    return normalized


def _normalize_references(
    references: Sequence[str] | Sequence[Sequence[str]],
    expected_length: int,
) -> list[list[str]]:
    if not references:
        raise ValueError("references must not be empty.")

    first_item = references[0]
    if isinstance(first_item, str):
        normalized = [[str(reference).strip()] for reference in references]
    else:
        normalized = [[str(reference).strip() for reference in sample] for sample in references]

    if len(normalized) != expected_length:
        raise ValueError("predictions and references must have the same length.")

    for sample_references in normalized:
        if not sample_references or any(not reference for reference in sample_references):
            raise ValueError("each sample must include at least one non-empty reference string.")

    return normalized


def _load_evaluate(metric_name: str) -> Any:
    try:
        evaluate = import_module("evaluate")
    except ImportError as exc:
        install_hint = "uv add evaluate" if metric_name == "bleu" else "uv add evaluate nltk"
        raise ImportError(
            f"{metric_name} requires the `evaluate` package. Install it with `{install_hint}`."
        ) from exc

    try:
        if metric_name == "meteor":
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                return evaluate.load(metric_name)
        return evaluate.load(metric_name)
    except ImportError as exc:
        raise ImportError(
            f"{metric_name} requires additional dependencies. Install them with `uv add evaluate nltk`."
        ) from exc


def _to_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _compute_evaluate_metric(
    metric_name: str,
    *,
    predictions: list[str],
    references: list[list[str]],
) -> dict[str, Any]:
    metric = _load_evaluate(metric_name)
    if metric_name == "meteor":
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return metric.compute(predictions=predictions, references=references)
    return metric.compute(predictions=predictions, references=references)


def _extract_ngrams(text: str, max_n: int) -> Counter[tuple[str, ...]]:
    tokens = text.split()
    counts: Counter[tuple[str, ...]] = Counter()
    for ngram_size in range(1, max_n + 1):
        for index in range(len(tokens) - ngram_size + 1):
            counts[tuple(tokens[index : index + ngram_size])] += 1
    return counts


def _counter_to_tfidf_vector(
    counts: Counter[tuple[str, ...]],
    *,
    doc_frequencies: Counter[tuple[str, ...]],
    log_reference_count: float,
    max_n: int,
) -> tuple[list[defaultdict[tuple[str, ...], float]], list[float], int]:
    vectors = [defaultdict(float) for _ in range(max_n)]
    norms = [0.0] * max_n
    length = 0

    for ngram, term_frequency in counts.items():
        ngram_index = len(ngram) - 1
        log_df = math.log(max(1.0, float(doc_frequencies[ngram])))
        tfidf = float(term_frequency) * (log_reference_count - log_df)
        vectors[ngram_index][ngram] = tfidf
        norms[ngram_index] += tfidf * tfidf
        if ngram_index == 1:
            length += term_frequency

    return vectors, [math.sqrt(norm) for norm in norms], length


def _similarity(
    candidate_vector: list[defaultdict[tuple[str, ...], float]],
    reference_vector: list[defaultdict[tuple[str, ...], float]],
    candidate_norms: list[float],
    reference_norms: list[float],
    *,
    candidate_length: int,
    reference_length: int,
    max_n: int,
    sigma: float,
) -> list[float]:
    similarities = [0.0] * max_n
    delta = float(candidate_length - reference_length)
    gaussian_penalty = math.exp(-((delta**2) / (2.0 * (sigma**2))))

    for ngram_index in range(max_n):
        score = 0.0
        for ngram, candidate_value in candidate_vector[ngram_index].items():
            reference_value = reference_vector[ngram_index].get(ngram, 0.0)
            score += min(candidate_value, reference_value) * reference_value

        if candidate_norms[ngram_index] and reference_norms[ngram_index]:
            score /= candidate_norms[ngram_index] * reference_norms[ngram_index]
        similarities[ngram_index] = score * gaussian_penalty

    return similarities


def _compute_cider_d(
    predictions: list[str],
    references: list[list[str]],
    *,
    max_n: int = 4,
    sigma: float = 6.0,
    scale: float = 10.0,
) -> float:
    cooked_predictions = [_extract_ngrams(prediction, max_n) for prediction in predictions]
    cooked_references = [
        [_extract_ngrams(reference, max_n) for reference in sample_references]
        for sample_references in references
    ]

    doc_frequencies: Counter[tuple[str, ...]] = Counter()
    for sample_references in cooked_references:
        for ngram in {ngram for reference in sample_references for ngram in reference}:
            doc_frequencies[ngram] += 1

    log_reference_count = math.log(float(len(cooked_references)))
    sample_scores: list[float] = []
    for candidate_counts, sample_references in zip(cooked_predictions, cooked_references, strict=True):
        candidate_vector, candidate_norms, candidate_length = _counter_to_tfidf_vector(
            candidate_counts,
            doc_frequencies=doc_frequencies,
            log_reference_count=log_reference_count,
            max_n=max_n,
        )

        ngram_score_sums = [0.0] * max_n
        for reference_counts in sample_references:
            reference_vector, reference_norms, reference_length = _counter_to_tfidf_vector(
                reference_counts,
                doc_frequencies=doc_frequencies,
                log_reference_count=log_reference_count,
                max_n=max_n,
            )
            similarities = _similarity(
                candidate_vector,
                reference_vector,
                candidate_norms,
                reference_norms,
                candidate_length=candidate_length,
                reference_length=reference_length,
                max_n=max_n,
                sigma=sigma,
            )
            for index, similarity in enumerate(similarities):
                ngram_score_sums[index] += similarity

        mean_similarity = sum(ngram_score_sums) / (len(sample_references) * max_n)
        sample_scores.append(mean_similarity * scale)

    return sum(sample_scores) / len(sample_scores)


def compute_caption_metrics(
    predictions: list[str],
    references: list[str] | list[list[str]],
    metrics: Iterable[str] = ("bleu", "meteor", "cider_d"),
) -> dict[str, float]:
    normalized_predictions = _normalize_predictions(predictions)
    normalized_references = _normalize_references(references, len(normalized_predictions))

    scores: dict[str, float] = {}
    for metric_name in metrics:
        if metric_name not in SUPPORTED_METRICS:
            raise ValueError(f"unsupported metric: {metric_name}")

        if metric_name in {"bleu", "meteor"}:
            result = _compute_evaluate_metric(
                metric_name,
                predictions=normalized_predictions,
                references=normalized_references,
            )
            scores[metric_name] = _to_float(result[metric_name])
            continue

        scores["cider_d"] = _compute_cider_d(normalized_predictions, normalized_references)

    return scores


__all__ = ["compute_caption_metrics"]
