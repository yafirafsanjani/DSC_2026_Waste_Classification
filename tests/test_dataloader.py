import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import SequentialSampler

from src.data.dataloader import ManifestImageDataset, ManifestValidationError, create_dataloader


def _make_project(tmp_path: Path) -> Path:
    shutil.copytree(Path.cwd() / "configs", tmp_path / "configs")
    rows = []
    for split in ("train", "val", "test_internal"):
        for index, name in enumerate(("alpha", "beta")):
            filename = f"{split}_{name}.png"
            processed = tmp_path / "data" / "processed" / split / name / filename
            processed.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((12, 15, 3), index * 100, dtype=np.uint8)).save(processed)
            rows.append({"class_name": name, "copy_sha256_verified": "True", "filename": filename,
                         "group_id": f"g-{split}-{name}", "perceptual_hash": "x",
                         "processed_relative_path": processed.relative_to(tmp_path).as_posix(), "sha256": f"hash-{split}-{name}",
                         "source_identity": f"source-{split}-{name}",
                         "source_relative_path": f"data/extracted/v1/train/{name}/{filename}", "split": split})
    manifest = tmp_path / "data" / "metadata" / "split_manifest.csv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def _tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255


def test_manifest_dataset_mapping_and_read_only_files(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    image_path = next((root / "data" / "processed" / "train").rglob("*.png"))
    before = image_path.read_bytes()
    dataset = ManifestImageDataset("train", project_root=root, transform=_tensor)
    tensor, target = dataset[0]
    assert dataset.class_to_idx == {"alpha": 0, "beta": 1}
    assert tensor.shape == (3, 12, 15) and target in {0, 1}
    assert image_path.read_bytes() == before


def test_loader_shuffle_is_seeded_and_validation_is_sequential(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    first = create_dataloader("train", project_root=root, batch_size=1, seed=42, transform=_tensor, pin_memory=False)
    second = create_dataloader("train", project_root=root, batch_size=1, seed=42, transform=_tensor, pin_memory=False)
    assert [int(label) for _, labels in first for label in labels] == [int(label) for _, labels in second for label in labels]
    validation = create_dataloader("val", project_root=root, batch_size=1, transform=_tensor, pin_memory=False)
    assert isinstance(validation.sampler, SequentialSampler)
    with pytest.raises(ValueError, match="must not shuffle"):
        create_dataloader("val", project_root=root, shuffle=True, transform=_tensor, pin_memory=False)


def test_forbidden_source_is_rejected(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    manifest = root / "data" / "metadata" / "split_manifest.csv"
    text = manifest.read_text(encoding="utf-8").replace("data/extracted/v1/train/alpha", "data/extracted/v5/alpha", 1)
    manifest.write_text(text, encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="Forbidden source"):
        ManifestImageDataset("train", project_root=root, transform=_tensor)
