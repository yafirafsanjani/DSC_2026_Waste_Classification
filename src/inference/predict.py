"""Read-only checkpoint loading and reusable PyTorch classifier prediction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn

from src.callbacks.model_checkpoint import load_checkpoint
from src.data.preprocessing import load_active_class_mapping
from src.models import CNNFromScratch


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot safely be used for the requested model."""


@dataclass(frozen=True)
class Prediction:
    """One predicted category aligned to its immutable competition image ID."""

    image_id: str
    waste_category: str


@dataclass(frozen=True)
class LoadedTorchClassifier:
    """A classifier restored only for evaluation and its checkpoint-derived labels."""

    model: nn.Module
    class_mapping: dict[str, int]
    index_to_class: tuple[str, ...]
    device: torch.device
    checkpoint_path: Path
    checkpoint_epoch: int | None
    best_epoch: int | None
    best_metric: float | None


def resolve_inference_device(device: torch.device | str | None = None) -> torch.device:
    """Prefer CUDA when available unless a concrete device was requested."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validated_class_mapping(class_mapping: Mapping[str, object]) -> dict[str, int]:
    if not class_mapping:
        raise CheckpointCompatibilityError("Checkpoint class mapping is empty.")
    normalized: dict[str, int] = {}
    for class_name, index in class_mapping.items():
        if not isinstance(class_name, str) or not class_name:
            raise CheckpointCompatibilityError("Checkpoint class mapping contains an invalid class name.")
        if not isinstance(index, int):
            raise CheckpointCompatibilityError("Checkpoint class mapping indices must be integers.")
        normalized[class_name] = index
    if set(normalized.values()) != set(range(len(normalized))):
        raise CheckpointCompatibilityError("Checkpoint class mapping indices must be contiguous from zero.")
    return normalized


def load_cnn_best_checkpoint(
    *,
    project_root: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    device: torch.device | str | None = None,
    expected_class_mapping: Mapping[str, int] | None = None,
) -> LoadedTorchClassifier:
    """Load CNN best weights without optimizer, training, or checkpoint mutation."""
    root = Path(project_root or Path.cwd()).resolve()
    selected_checkpoint = Path(checkpoint_path or root / "models" / "checkpoints" / "cnn_scratch_baseline_v1" / "best.pt")
    selected_device = resolve_inference_device(device)
    payload = load_checkpoint(selected_checkpoint, map_location=selected_device)
    if payload.get("model_identifier") != "cnn_from_scratch":
        raise CheckpointCompatibilityError("Checkpoint model identifier is not compatible with CNNFromScratch.")
    class_mapping = _validated_class_mapping(payload.get("class_mapping", {}))
    active_mapping = dict(expected_class_mapping) if expected_class_mapping is not None else load_active_class_mapping(root)
    if class_mapping != active_mapping:
        raise CheckpointCompatibilityError("Checkpoint class mapping does not match the active training mapping.")
    dropout = float(payload.get("training_config", {}).get("dropout", 0.40))
    model = CNNFromScratch(num_classes=len(class_mapping), dropout=dropout)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(selected_device)
    model.eval()
    index_to_class = tuple(class_name for class_name, _ in sorted(class_mapping.items(), key=lambda item: item[1]))
    return LoadedTorchClassifier(
        model=model,
        class_mapping=class_mapping,
        index_to_class=index_to_class,
        device=selected_device,
        checkpoint_path=selected_checkpoint,
        checkpoint_epoch=payload.get("epoch"),
        best_epoch=payload.get("best_epoch"),
        best_metric=payload.get("best_metric"),
    )


def predict_torch_classifier(
    model: nn.Module,
    dataloader: Iterable[tuple[torch.Tensor, Sequence[str]]],
    *,
    index_to_class: Sequence[str],
    device: torch.device | str | None = None,
) -> list[Prediction]:
    """Predict an unlabeled sequential loader in eval/inference mode only."""
    selected_device = resolve_inference_device(device)
    if not index_to_class or len(set(index_to_class)) != len(index_to_class):
        raise ValueError("index_to_class must contain unique non-empty class names.")
    model.to(selected_device)
    model.eval()
    predictions: list[Prediction] = []
    with torch.inference_mode():
        for images, image_ids in dataloader:
            if not isinstance(images, torch.Tensor):
                raise TypeError("Inference loader must yield image tensors as its first value.")
            identifiers = tuple(str(image_id) for image_id in image_ids)
            if images.shape[0] != len(identifiers):
                raise ValueError("Inference image batch and image ID batch have different lengths.")
            logits = model(images.to(selected_device, non_blocking=True))
            if logits.ndim != 2 or logits.shape != (len(identifiers), len(index_to_class)):
                raise ValueError("Classifier logits do not match the expected batch size or class mapping.")
            for image_id, class_index in zip(identifiers, logits.argmax(dim=1).cpu().tolist()):
                predictions.append(Prediction(image_id=image_id, waste_category=index_to_class[int(class_index)]))
    return predictions
