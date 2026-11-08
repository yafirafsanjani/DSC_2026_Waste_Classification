"""Manifest-backed, split-safe PyTorch datasets and data loaders."""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.augmentation import build_split_augmentation
from src.data.preprocessing import (
    ConfigValidationError,
    ModelAwareTransform,
    build_transforms,
    coerce_rgb_image,
    load_active_class_mapping,
    resolve_model_config,
)

VALID_SPLITS = {"train", "val", "test_internal"}
ALLOWED_SOURCE_PREFIX = "data/extracted/v1/train/"


@dataclass(frozen=True)
class ManifestRecord:
    class_name: str
    filename: str
    split: str
    source_relative_path: str
    processed_relative_path: str
    sha256: str
    source_identity: str
    group_id: str


class ManifestValidationError(ValueError):
    """Raised when active dataset metadata no longer satisfies its contract."""


def _project_root(project_root: Path | str | None) -> Path:
    return Path(project_root or Path.cwd()).resolve()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ManifestValidationError(f"Split manifest does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"class_name", "filename", "split", "source_relative_path", "processed_relative_path", "sha256", "source_identity", "group_id"}
    missing = required.difference(rows[0] if rows else set())
    if missing:
        raise ManifestValidationError(f"Split manifest is missing columns: {sorted(missing)}")
    return rows


def load_manifest_records(split: str, project_root: Path | str | None = None, manifest_path: Path | str | None = None, verify_files: bool = True) -> list[ManifestRecord]:
    """Load only one active split and reject forbidden data sources."""
    if split not in VALID_SPLITS:
        raise ManifestValidationError(f"Unsupported split: {split}")
    root = _project_root(project_root)
    path = Path(manifest_path) if manifest_path else root / "data" / "metadata" / "split_manifest.csv"
    rows = [row for row in _read_manifest(path) if row["split"] == split]
    if not rows:
        raise ManifestValidationError(f"Split manifest has no records for split: {split}")
    records, seen_sources = [], set()
    for row in rows:
        source = row["source_relative_path"].replace("\\", "/")
        processed = row["processed_relative_path"].replace("\\", "/")
        prefix = f"data/processed/{split}/{row['class_name']}/"
        if not source.startswith(ALLOWED_SOURCE_PREFIX):
            raise ManifestValidationError(f"Forbidden source path in active manifest: {source}")
        if not processed.startswith(prefix):
            raise ManifestValidationError(f"Invalid processed destination for {source}: {processed}")
        if source in seen_sources:
            raise ManifestValidationError(f"Duplicate source path in {split}: {source}")
        seen_sources.add(source)
        if verify_files and not (root / processed).is_file():
            raise ManifestValidationError(f"Processed image referenced by manifest is missing: {root / processed}")
        records.append(ManifestRecord(row["class_name"], row["filename"], split, source, processed, row["sha256"], row["source_identity"], row["group_id"]))
    return sorted(records, key=lambda record: record.source_relative_path)


def seed_worker(worker_id: int) -> None:
    del worker_id
    random.seed(torch.initial_seed() % 2**32)
    np.random.seed(torch.initial_seed() % 2**32)


class ManifestImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Read-only image dataset backed by the leakage-safe split manifest."""

    def __init__(self, split: str, model_type: str = "cnn", *, project_root: Path | str | None = None,
                 manifest_path: Path | str | None = None, model_config: Mapping[str, Any] | None = None,
                 transform: Callable[[Image.Image], torch.Tensor] | None = None,
                 augmentation_enabled: bool | None = None, seed: int = 42,
                 verify_files: bool = True) -> None:
        self.root = _project_root(project_root)
        self.split = split
        self.records = load_manifest_records(split, self.root, manifest_path, verify_files)
        self.class_to_idx = load_active_class_mapping(self.root)
        unknown = {record.class_name for record in self.records}.difference(self.class_to_idx)
        if unknown:
            raise ManifestValidationError(f"Manifest contains unmapped classes: {sorted(unknown)}")
        resolved = resolve_model_config(model_type, model_config, self.root)
        augmentation_config = dict(resolved.get("augmentation", {}))
        active_augmentation = bool(augmentation_config.get("enabled", False))
        if augmentation_enabled is not None:
            active_augmentation = augmentation_enabled
        augmentation = build_split_augmentation(
            split, str(augmentation_config.get("profile", "disabled")), active_augmentation, seed, self.root
        )
        self.transform = transform or build_transforms(
            model_type, split, config=model_config, project_root=self.root, augmentation=augmentation
        )
        if split != "train" and isinstance(self.transform, ModelAwareTransform) and self.transform.augmentation is not None:
            raise ConfigValidationError("Validation and internal-test transforms must not contain augmentation.")

    @property
    def classes(self) -> list[str]:
        return [name for name, _ in sorted(self.class_to_idx.items(), key=lambda item: item[1])]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image_path = self.root / record.processed_relative_path
        try:
            return self.transform(coerce_rgb_image(image_path)), self.class_to_idx[record.class_name]
        except Exception as error:
            raise RuntimeError(f"Unable to load active image {image_path}: {error}") from error


def create_dataloader(
    split: str, model_type: str = "cnn", *, batch_size: int = 32, shuffle: bool | None = None,
    num_workers: int = 0, pin_memory: bool | None = None, persistent_workers: bool = False,
    seed: int = 42, project_root: Path | str | None = None, manifest_path: Path | str | None = None,
    model_config: Mapping[str, Any] | None = None, transform: Callable[[Image.Image], torch.Tensor] | None = None,
    augmentation_enabled: bool | None = None, verify_files: bool = True,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a reproducible loader; only train defaults to shuffled batches."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative.")
    if split not in VALID_SPLITS:
        raise ManifestValidationError(f"Unsupported split: {split}")
    dataset = ManifestImageDataset(
        split, model_type, project_root=project_root, manifest_path=manifest_path, model_config=model_config,
        transform=transform, augmentation_enabled=augmentation_enabled, seed=seed, verify_files=verify_files,
    )
    use_shuffle = split == "train" if shuffle is None else bool(shuffle)
    if split != "train" and use_shuffle:
        raise ValueError("Validation and internal-test DataLoaders must not shuffle.")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=use_shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None, generator=generator,
    )
