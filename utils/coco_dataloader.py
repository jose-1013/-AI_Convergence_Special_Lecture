from __future__ import annotations

import json
import pickle
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .logger_utils import get_logger

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_CACHE_DIR = DEFAULT_DATA_DIR / "hf_cache"
DEFAULT_SUBSET_DIR = DEFAULT_DATA_DIR / "coco_subsets"
DEFAULT_DATASET_NAME = "jxie/coco_captions"
DEFAULT_SPLIT = "train"
DEFAULT_SEED = 42
DEFAULT_BUFFER_SIZE = 10_000
DEFAULT_MAX_SAMPLES = 5_000

CocoSample = dict[str, Any]
CocoImageSample = dict[str, Any]
LOGGER = get_logger(__name__)


def _require_datasets_load_dataset() -> Callable[..., Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install it with "
            "'uv add datasets' or 'uv run --with datasets ...'."
        ) from exc
    return load_dataset


def _progress(
    iterable: Iterable[Any],
    *,
    enabled: bool,
    desc: str,
) -> Iterable[Any]:
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        LOGGER.info("%s", desc)
        return iterable
    return tqdm(iterable, desc=desc)


def _set_progress_postfix(progress: Iterable[Any], **kwargs: Any) -> None:
    set_postfix = getattr(progress, "set_postfix", None)
    if callable(set_postfix):
        set_postfix(**kwargs)


def _normalize_caption(caption: Any) -> str:
    if isinstance(caption, str):
        return caption.strip()
    if isinstance(caption, Sequence) and not isinstance(caption, (bytes, bytearray, str)):
        for item in caption:
            text = str(item).strip()
            if text:
                return text
        return ""
    return str(caption).strip()


def normalize_coco_sample(sample: Mapping[str, Any]) -> CocoSample:
    return {
        "image": sample["image"],
        "caption": _normalize_caption(sample["caption"]),
        "cocoid": int(sample["cocoid"]),
        "filename": str(sample["filename"]),
    }


def _dataset_slug(dataset_name: str) -> str:
    return dataset_name.replace("/", "__")


def _subset_manifest_path(
    *,
    dataset_name: str,
    split: str,
    max_images: int,
    shuffle: bool,
    seed: int,
    buffer_size: int,
) -> Path:
    mode = "shuffle" if shuffle else "ordered"
    filename = (
        f"{_dataset_slug(dataset_name)}_{split}_{mode}"
        f"_seed{seed}_buffer{buffer_size}_max{max_images}.json"
    )
    return DEFAULT_SUBSET_DIR / filename


def _load_subset_ids(manifest_path: Path) -> list[int] | None:
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or "cocoids" not in payload:
        return None
    return [int(cocoid) for cocoid in payload["cocoids"]]


def _save_subset_ids(manifest_path: Path, cocoids: Sequence[int]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"version": 1, "cocoids": list(cocoids)}, indent=2),
        encoding="utf-8",
    )


def _subset_payload_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(".pkl")


def _load_subset_payload(manifest_path: Path) -> list[CocoImageSample] | None:
    payload_path = _subset_payload_path(manifest_path)
    if not payload_path.is_file():
        return None
    with payload_path.open("rb") as file:
        return pickle.load(file)


def _save_subset_payload(manifest_path: Path, samples: Sequence[CocoImageSample]) -> None:
    payload_path = _subset_payload_path(manifest_path)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    with payload_path.open("wb") as file:
        pickle.dump(list(samples), file, protocol=pickle.HIGHEST_PROTOCOL)


def _group_coco_images(
    samples: Iterable[CocoSample],
    *,
    max_images: int,
    show_progress: bool,
    allow_early_stop: bool,
) -> list[CocoImageSample]:
    grouped_by_id: dict[int, CocoImageSample] = {}
    progress = _progress(
        samples,
        enabled=show_progress,
        desc="Grouping COCO images",
    )
    _set_progress_postfix(progress, images=0, target=max_images)
    for sample in progress:
        cocoid = int(sample["cocoid"])
        grouped_sample = grouped_by_id.get(cocoid)
        if grouped_sample is None:
            grouped_sample = {
                "image": sample["image"],
                "caption": sample["caption"],
                "captions": [sample["caption"]],
                "cocoid": cocoid,
                "filename": sample["filename"],
            }
            grouped_by_id[cocoid] = grouped_sample
            _set_progress_postfix(
                progress,
                images=len(grouped_by_id),
                target=max_images,
            )
            if allow_early_stop and len(grouped_by_id) >= max_images:
                break
        else:
            grouped_sample["captions"].append(sample["caption"])
    return list(grouped_by_id.values())


def _collect_saved_subset_samples(
    *,
    selected_ids: Sequence[int],
    split: str,
    dataset_name: str,
    streaming: bool,
    show_progress: bool,
    seed: int,
    buffer_size: int,
    **load_dataset_kwargs: Any,
) -> list[CocoImageSample]:
    selected_id_set = set(selected_ids)
    grouped_by_id: dict[int, CocoImageSample] = {}
    samples = iter_coco_samples(
        split=split,
        dataset_name=dataset_name,
        streaming=streaming,
        shuffle=False,
        seed=seed,
        buffer_size=buffer_size,
        **load_dataset_kwargs,
    )
    progress = _progress(
        samples,
        enabled=show_progress,
        desc="Collecting COCO images from saved manifest",
    )
    _set_progress_postfix(progress, images=0, target=len(selected_ids))
    for sample in progress:
        cocoid = int(sample["cocoid"])
        if cocoid not in selected_id_set:
            if len(grouped_by_id) >= len(selected_ids):
                break
            continue
        grouped_sample = grouped_by_id.get(cocoid)
        if grouped_sample is None:
            grouped_by_id[cocoid] = {
                "image": sample["image"],
                "caption": sample["caption"],
                "captions": [sample["caption"]],
                "cocoid": cocoid,
                "filename": sample["filename"],
            }
            _set_progress_postfix(
                progress,
                images=len(grouped_by_id),
                target=len(selected_ids),
            )
        else:
            grouped_sample["captions"].append(sample["caption"])

    return [grouped_by_id[cocoid] for cocoid in selected_ids if cocoid in grouped_by_id]


def sample_coco_image_subset(
    max_images: int = DEFAULT_MAX_SAMPLES,
    *,
    split: str = DEFAULT_SPLIT,
    dataset_name: str = DEFAULT_DATASET_NAME,
    streaming: bool = True,
    shuffle: bool = True,
    show_progress: bool = False,
    seed: int = DEFAULT_SEED,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    **load_dataset_kwargs: Any,
) -> list[CocoImageSample]:
    if max_images <= 0:
        raise ValueError("max_images must be a positive integer.")

    manifest_path = _subset_manifest_path(
        dataset_name=dataset_name,
        split=split,
        max_images=max_images,
        shuffle=shuffle,
        seed=seed,
        buffer_size=buffer_size,
    )
    selected_ids = _load_subset_ids(manifest_path)
    saved_samples = _load_subset_payload(manifest_path)

    if saved_samples is not None:
        LOGGER.info("Using saved COCO subset payload: %s", _subset_payload_path(manifest_path))
        return list(saved_samples)

    if selected_ids is None:
        samples = iter_coco_samples(
            split=split,
            dataset_name=dataset_name,
            streaming=streaming,
            shuffle=shuffle,
            seed=seed,
            buffer_size=buffer_size,
            **load_dataset_kwargs,
        )
        grouped_samples = _group_coco_images(
            samples,
            max_images=max_images,
            show_progress=show_progress,
            allow_early_stop=True,
        )
        grouped_samples = grouped_samples[:max_images]
        selected_ids = [int(sample["cocoid"]) for sample in grouped_samples]
        _save_subset_ids(manifest_path, selected_ids)
        _save_subset_payload(manifest_path, grouped_samples)
        return grouped_samples

    LOGGER.info("Using saved COCO subset manifest: %s", manifest_path)
    grouped_samples = _collect_saved_subset_samples(
        selected_ids=selected_ids,
        split=split,
        dataset_name=dataset_name,
        streaming=streaming,
        show_progress=show_progress,
        seed=seed,
        buffer_size=buffer_size,
        **load_dataset_kwargs,
    )
    _save_subset_payload(manifest_path, grouped_samples)
    return grouped_samples


def expand_coco_image_samples(samples: Sequence[CocoImageSample]) -> list[CocoSample]:
    expanded_samples: list[CocoSample] = []
    for sample in samples:
        for caption in sample["captions"]:
            expanded_samples.append(
                {
                    "image": sample["image"],
                    "caption": caption,
                    "cocoid": sample["cocoid"],
                    "filename": sample["filename"],
                }
            )
    return expanded_samples


class COCOSubsetDataset:
    def __init__(
        self,
        samples: Sequence[CocoSample],
        *,
        image_transform: Callable[[Any], Any] | None = None,
        caption_transform: Callable[[str], Any] | None = None,
    ) -> None:
        self.samples = list(samples)
        self.image_transform = image_transform
        self.caption_transform = caption_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CocoSample:
        sample = dict(self.samples[index])
        if self.image_transform is not None:
            sample["image"] = self.image_transform(sample["image"])
        if self.caption_transform is not None:
            sample["caption"] = self.caption_transform(sample["caption"])
        return sample


def load_coco_dataset(
    split: str = DEFAULT_SPLIT,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    streaming: bool = True,
    **load_dataset_kwargs: Any,
) -> Any:
    load_dataset = _require_datasets_load_dataset()
    cache_dir = load_dataset_kwargs.pop("cache_dir", DEFAULT_CACHE_DIR)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
        **load_dataset_kwargs,
    )


def iter_coco_samples(
    split: str = DEFAULT_SPLIT,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    streaming: bool = True,
    shuffle: bool = True,
    seed: int = DEFAULT_SEED,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    **load_dataset_kwargs: Any,
) -> Iterable[CocoSample]:
    dataset = load_coco_dataset(
        split=split,
        dataset_name=dataset_name,
        streaming=streaming,
        **load_dataset_kwargs,
    )

    if shuffle:
        if streaming:
            dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
        else:
            dataset = dataset.shuffle(seed=seed)

    for sample in dataset:
        yield normalize_coco_sample(sample)

def coco_collate_fn(
    batch: Sequence[CocoSample],
) -> dict[str, list[Any]]:
    images = [sample["image"] for sample in batch]
    captions = [sample["caption"] for sample in batch]

    return {
        "images": images,
        "captions": captions,
        "cocoids": [sample["cocoid"] for sample in batch],
        "filenames": [sample["filename"] for sample in batch],
        "samples": list(batch),
    }


def build_coco_dataloader(
    batch_size: int,
    *,
    split: str = DEFAULT_SPLIT,
    dataset_name: str = DEFAULT_DATASET_NAME,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    streaming: bool = True,
    sample_shuffle: bool = True,
    dataloader_shuffle: bool = False,
    seed: int = DEFAULT_SEED,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    image_transform: Callable[[Any], Any] | None = None,
    caption_transform: Callable[[str], Any] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    **load_dataset_kwargs: Any,
) -> Any:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer.")

    samples: list[CocoSample] = []
    for index, sample in enumerate(
        iter_coco_samples(
            split=split,
            dataset_name=dataset_name,
            streaming=streaming,
            shuffle=sample_shuffle,
            seed=seed,
            buffer_size=buffer_size,
            **load_dataset_kwargs,
        )
    ):
        if index >= max_samples:
            break
        samples.append(sample)

    dataset = create_coco_dataset(
        samples,
        image_transform=image_transform,
        caption_transform=caption_transform,
    )

    return create_coco_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_coco_dataset(
    samples: Sequence[CocoSample],
    *,
    image_transform: Callable[[Any], Any] | None = None,
    caption_transform: Callable[[str], Any] | None = None,
) -> COCOSubsetDataset:
    return COCOSubsetDataset(
        samples,
        image_transform=image_transform,
        caption_transform=caption_transform,
    )


def create_coco_dataloader(
    dataset: COCOSubsetDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> Any:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=coco_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DATASET_NAME",
    "DEFAULT_DATA_DIR",
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_SEED",
    "DEFAULT_SPLIT",
    "DEFAULT_SUBSET_DIR",
    "CocoImageSample",
    "COCOSubsetDataset",
    "build_coco_dataloader",
    "coco_collate_fn",
    "create_coco_dataset",
    "create_coco_dataloader",
    "expand_coco_image_samples",
    "iter_coco_samples",
    "load_coco_dataset",
    "normalize_coco_sample",
    "sample_coco_image_subset",
]
