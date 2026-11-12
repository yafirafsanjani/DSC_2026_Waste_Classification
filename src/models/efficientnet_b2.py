"""EfficientNet-B2 ImageNet transfer-learning helpers for the V1 experiment."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2
from torchvision.transforms import InterpolationMode

from src.callbacks.model_checkpoint import load_checkpoint
from src.data.preprocessing import load_active_class_mapping, load_yaml
from src.inference.predict import LoadedTorchClassifier, resolve_inference_device
from src.training.fine_tuning import build_parameter_groups

MODEL_IDENTIFIER = "efficientnet_b2_imagenet1k_v1"
WEIGHTS_NAME = "EfficientNet_B2_Weights.IMAGENET1K_V1"
HEAD_EXPERIMENT_ID = "efficientnet_b2_head_v1"
FINETUNE_EXPERIMENT_ID = "efficientnet_b2_finetune_v1"
SELECTION_FILENAME = "efficientnet_b2_selected_v1.json"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 288


class EfficientNetB2Error(RuntimeError):
    """Raised when the EfficientNet-B2 experiment contract is violated."""


class PretrainedWeightsUnavailableError(EfficientNetB2Error):
    """Raised instead of silently replacing requested ImageNet weights with random weights."""


def get_efficientnet_b2_weights() -> EfficientNet_B2_Weights:
    return EfficientNet_B2_Weights.IMAGENET1K_V1


def build_efficientnet_b2(num_classes: int, *, pretrained: bool) -> nn.Module:
    """Build the eight-class classifier while retaining torchvision's dropout layer."""
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one.")
    weights = get_efficientnet_b2_weights() if pretrained else None
    try:
        model = efficientnet_b2(weights=weights)
    except Exception as error:
        if pretrained:
            raise PretrainedWeightsUnavailableError(
                "EfficientNet-B2 ImageNet-1K weights could not be loaded. Connect to the internet once "
                "to download torchvision's EfficientNet_B2_Weights.IMAGENET1K_V1 into the Torch Hub "
                "cache, then rerun this cell. Random initialization is intentionally not used. "
                f"Original error: {error}"
            ) from error
        raise
    if not isinstance(model.classifier, nn.Sequential) or not isinstance(model.classifier[-1], nn.Linear):
        raise EfficientNetB2Error("Unexpected torchvision EfficientNet-B2 classifier structure.")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def model_summary(model: nn.Module, *, batch_size: int = 2, device: torch.device | str = "cpu") -> dict[str, Any]:
    selected_device = torch.device(device)
    model.to(selected_device).eval()
    with torch.inference_mode():
        output = model(torch.zeros(batch_size, 3, INPUT_SIZE, INPUT_SIZE, device=selected_device))
    if tuple(output.shape) != (batch_size, 8):
        raise EfficientNetB2Error(f"Expected classifier output ({batch_size}, 8), got {tuple(output.shape)}.")
    return {
        "architecture": "torchvision.models.efficientnet_b2",
        "pretrained_weights": WEIGHTS_NAME,
        "parameter_counts": count_parameters(model),
        "classifier": str(model.classifier),
        "output_shape": tuple(output.shape),
        "device": str(selected_device),
    }


def build_efficientnet_b2_transforms() -> tuple[transforms.Compose, Any, dict[str, Any]]:
    """Return light train augmentation and official deterministic validation preprocessing."""
    weights = get_efficientnet_b2_weights()
    validation_transform = weights.transforms()
    resize_size = tuple(int(value) for value in validation_transform.resize_size)
    crop_size = tuple(int(value) for value in validation_transform.crop_size)
    if resize_size != (INPUT_SIZE,) or crop_size != (INPUT_SIZE,):
        raise EfficientNetB2Error(
            f"Unexpected torchvision B2 preprocessing: resize={resize_size}, crop={crop_size}."
        )
    if tuple(validation_transform.mean) != IMAGENET_MEAN or tuple(validation_transform.std) != IMAGENET_STD:
        raise EfficientNetB2Error("EfficientNet-B2 ImageNet normalization metadata has changed.")
    if validation_transform.interpolation != InterpolationMode.BICUBIC:
        raise EfficientNetB2Error("EfficientNet-B2 official transform must use bicubic interpolation.")
    training_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, shear=(-10, 10), interpolation=InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=0.10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    metadata = {
        "source": WEIGHTS_NAME,
        "color_mode": "RGB",
        "resize": INPUT_SIZE,
        "center_crop": INPUT_SIZE,
        "interpolation": "bicubic",
        "pixel_scaling": "[0, 1] before normalization",
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "training_augmentation": {
            "horizontal_flip": True,
            "rotation_degrees": 15,
            "shear_degrees": 10,
            "brightness": 0.10,
        },
    }
    return training_transform, validation_transform, metadata


def validate_preprocessing(transform: Any) -> dict[str, Any]:
    """Smoke-check the official deterministic transform without reading project data."""
    tensor = transform(Image.new("RGB", (401, 257), (128, 64, 32)))
    if not isinstance(tensor, torch.Tensor) or tensor.shape != (3, INPUT_SIZE, INPUT_SIZE):
        raise EfficientNetB2Error("Validation preprocessing does not produce CHW 288x288 tensors.")
    if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
        raise EfficientNetB2Error("Validation preprocessing must produce finite float32 tensors.")
    return {"shape": tuple(tensor.shape), "dtype": str(tensor.dtype), "finite": bool(torch.isfinite(tensor).all())}


def apply_batch_norm_policy(model: nn.Module, *, trainable_batch_norm: bool) -> None:
    """Keep frozen BatchNorm layers in eval mode after every model.train() call."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module_trainable = any(parameter.requires_grad for parameter in module.parameters())
            if not trainable_batch_norm or not module_trainable:
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False


def freeze_for_head_training(model: nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    apply_batch_norm_policy(model, trainable_batch_norm=False)


def configure_partial_fine_tuning(model: nn.Module, *, unfreeze_last_feature_modules: int = 2) -> list[str]:
    """Unfreeze the final MBConv stage and final convolution, not the full backbone."""
    if unfreeze_last_feature_modules < 1:
        raise ValueError("unfreeze_last_feature_modules must be positive.")
    feature_modules = list(model.features.children())
    if len(feature_modules) < unfreeze_last_feature_modules:
        raise EfficientNetB2Error("EfficientNet features are shorter than the requested unfreeze range.")
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for module in feature_modules[-unfreeze_last_feature_modules:]:
        for parameter in module.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    apply_batch_norm_policy(model, trainable_batch_norm=True)
    first_unfrozen = len(feature_modules) - unfreeze_last_feature_modules
    return [f"features.{index}" for index in range(first_unfrozen, len(feature_modules))]


def build_head_optimizer(model: nn.Module, *, learning_rate: float = 1e-3, weight_decay: float = 1e-4) -> torch.optim.AdamW:
    parameters = [parameter for parameter in model.classifier.parameters() if parameter.requires_grad]
    if not parameters:
        raise EfficientNetB2Error("Classifier head has no trainable parameters.")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def build_fine_tuning_optimizer(
    model: nn.Module, *, backbone_learning_rate: float = 1e-5,
    classifier_learning_rate: float = 1e-4, weight_decay: float = 1e-4,
) -> torch.optim.AdamW:
    groups = build_parameter_groups(
        model.features, model.classifier, backbone_lr=backbone_learning_rate,
        classifier_lr=classifier_learning_rate, weight_decay=weight_decay,
    )
    return torch.optim.AdamW(groups)


def head_training_config() -> dict[str, Any]:
    _, _, preprocessing = build_efficientnet_b2_transforms()
    return {
        "phase": "head", "architecture": "efficientnet_b2", "model_identifier": MODEL_IDENTIFIER,
        "pretrained_weights": WEIGHTS_NAME, "preprocessing": preprocessing,
        "batch_norm_policy": "all backbone BatchNorm layers eval; no running-stat updates",
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "gradient_clip_norm": 1.0,
        "max_epochs": 15, "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 5},
    }


def fine_tuning_config() -> dict[str, Any]:
    _, _, preprocessing = build_efficientnet_b2_transforms()
    return {
        "phase": "finetune", "architecture": "efficientnet_b2", "model_identifier": MODEL_IDENTIFIER,
        "pretrained_weights": WEIGHTS_NAME, "preprocessing": preprocessing,
        "unfreeze": "features[-2:] (final MBConv stage and final convolution)",
        "batch_norm_policy": "frozen BatchNorm eval; trainable final-stage BatchNorm may update",
        "optimizer": "AdamW", "backbone_learning_rate": 1e-5, "classifier_learning_rate": 1e-4,
        "weight_decay": 1e-4, "gradient_clip_norm": 1.0, "max_epochs": 45,
        "scheduler": {"name": "reduce_on_plateau", "monitor": "val_loss", "factor": 0.5, "patience": 4, "min_lr": 1e-7},
        "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 10},
    }


def ensure_experiment_is_safe_to_start(project_root: Path | str, experiment_id: str, *, resume: bool = False) -> None:
    root = Path(project_root).resolve()
    checkpoint_dir = root / "models" / "checkpoints" / experiment_id
    paths = (checkpoint_dir / "best.pt", checkpoint_dir / "last.pt",
             root / "outputs" / "metrics" / f"{experiment_id}.json",
             root / "outputs" / "metrics" / f"{experiment_id}.csv",
             root / "outputs" / "logs" / f"{experiment_id}.jsonl")
    existing = [path for path in paths if path.exists()]
    if existing and not resume:
        names = ", ".join(str(path.relative_to(root)) for path in existing)
        raise FileExistsError(
            f"Experiment {experiment_id!r} already has artifacts ({names}). Set its explicit resume flag "
            "or use a new experiment namespace; existing artifacts will not be overwritten."
        )


def validate_resume_configuration(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Permit explicit batch/accumulation/device changes after OOM while preserving model semantics."""
    mutable = {"batch_size", "gradient_accumulation_steps", "amp"}
    if {key: value for key, value in saved.items() if key not in mutable} != {key: value for key, value in current.items() if key not in mutable}:
        raise EfficientNetB2Error("Resume configuration differs beyond batch size, accumulation, or AMP.")


@dataclass(frozen=True)
class SelectedCheckpoint:
    phase: str
    experiment_id: str
    checkpoint_path: Path
    val_loss: float
    epoch: int | None
    class_mapping: dict[str, int]
    preprocessing: Mapping[str, Any]


def _phase_candidate(root: Path, phase: str, experiment_id: str) -> SelectedCheckpoint | None:
    path = root / "models" / "checkpoints" / experiment_id / "best.pt"
    if not path.is_file():
        return None
    payload = load_checkpoint(path, map_location="cpu")
    if payload.get("model_identifier") != MODEL_IDENTIFIER:
        raise EfficientNetB2Error(f"Checkpoint has incompatible model identifier: {path}")
    metric = payload.get("best_metric", payload.get("metric"))
    if metric is None or not np.isfinite(float(metric)) or payload.get("monitor") != "val_loss" or payload.get("mode") != "min":
        raise EfficientNetB2Error(f"Checkpoint has no validation-loss metric: {path}")
    config = dict(payload.get("training_config", {}))
    preprocessing = config.get("preprocessing")
    if config.get("phase") != phase or config.get("pretrained_weights") != WEIGHTS_NAME:
        raise EfficientNetB2Error(f"Checkpoint phase or pretrained provenance is incompatible: {path}")
    if not isinstance(preprocessing, Mapping):
        raise EfficientNetB2Error(f"Checkpoint has no reusable preprocessing metadata: {path}")
    return SelectedCheckpoint(
        phase=phase, experiment_id=experiment_id, checkpoint_path=path, val_loss=float(metric),
        epoch=payload.get("best_epoch", payload.get("epoch")),
        class_mapping={str(key): int(value) for key, value in dict(payload.get("class_mapping", {})).items()},
        preprocessing=preprocessing,
    )


def select_best_phase_checkpoint(project_root: Path | str) -> SelectedCheckpoint:
    root = Path(project_root).resolve()
    candidates = [candidate for candidate in (
        _phase_candidate(root, "head", HEAD_EXPERIMENT_ID),
        _phase_candidate(root, "finetune", FINETUNE_EXPERIMENT_ID),
    ) if candidate is not None]
    if not candidates:
        raise FileNotFoundError("No EfficientNet-B2 best checkpoint exists. Run Phase A manually first.")
    active_mapping = load_active_class_mapping(root)
    _, _, expected_preprocessing = build_efficientnet_b2_transforms()
    for candidate in candidates:
        if candidate.class_mapping != active_mapping or candidate.preprocessing != expected_preprocessing:
            raise EfficientNetB2Error(f"Checkpoint mapping or preprocessing does not match the active dataset: {candidate.checkpoint_path}")
    return min(candidates, key=lambda candidate: candidate.val_loss)


def save_selected_checkpoint(project_root: Path | str, selected: SelectedCheckpoint | None = None) -> Path:
    root = Path(project_root).resolve()
    choice = selected or select_best_phase_checkpoint(root)
    path = root / "models" / "checkpoints" / SELECTION_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "format_version": 1, "model_identifier": MODEL_IDENTIFIER, "phase": choice.phase,
        "experiment_id": choice.experiment_id, "checkpoint_path": str(choice.checkpoint_path.relative_to(root)),
        "val_loss": choice.val_loss, "epoch": choice.epoch, "class_mapping": choice.class_mapping,
        "preprocessing": choice.preprocessing,
    }, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


def load_selected_checkpoint(project_root: Path | str) -> SelectedCheckpoint:
    """Recompute the winner to avoid stale selection metadata after a later phase."""
    return select_best_phase_checkpoint(project_root)


def load_efficientnet_b2_checkpoint(
    *, project_root: Path | str, checkpoint_path: Path | str | None = None,
    device: torch.device | str | None = None, expected_class_mapping: Mapping[str, int] | None = None,
) -> LoadedTorchClassifier:
    """Load a final checkpoint with ``weights=None`` for fresh-kernel inference."""
    root = Path(project_root).resolve()
    choice = load_selected_checkpoint(root) if checkpoint_path is None else None
    selected_path = Path(checkpoint_path) if checkpoint_path is not None else choice.checkpoint_path
    selected_device = resolve_inference_device(device)
    payload = load_checkpoint(selected_path, map_location="cpu")
    if payload.get("model_identifier") != MODEL_IDENTIFIER:
        raise EfficientNetB2Error("Checkpoint is not an EfficientNet-B2 ImageNet-1K experiment artifact.")
    class_mapping = {str(key): int(value) for key, value in dict(payload.get("class_mapping", {})).items()}
    active_mapping = dict(expected_class_mapping) if expected_class_mapping is not None else load_active_class_mapping(root)
    if class_mapping != active_mapping or len(class_mapping) != 8 or set(class_mapping.values()) != set(range(8)):
        raise EfficientNetB2Error("Checkpoint class mapping does not match the active eight-class V1 manifest.")
    _, _, expected_preprocessing = build_efficientnet_b2_transforms()
    if payload.get("training_config", {}).get("preprocessing") != expected_preprocessing:
        raise EfficientNetB2Error("Checkpoint preprocessing metadata does not match the official B2 transform.")
    model = build_efficientnet_b2(len(class_mapping), pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(selected_device).eval()
    index_to_class = tuple(name for name, _ in sorted(class_mapping.items(), key=lambda item: item[1]))
    return LoadedTorchClassifier(
        model=model, class_mapping=class_mapping, index_to_class=index_to_class, device=selected_device,
        checkpoint_path=selected_path, checkpoint_epoch=payload.get("epoch"),
        best_epoch=payload.get("best_epoch"), best_metric=payload.get("best_metric"),
    )


def load_experiment_config(project_root: Path | str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_yaml(root / "configs" / "models" / "efficientnet.yaml")
    if config.get("model_type") != "efficientnet_b2" or config.get("weights") != WEIGHTS_NAME:
        raise EfficientNetB2Error("configs/models/efficientnet.yaml is not configured for EfficientNet-B2 IMAGENET1K_V1.")
    preprocessing = config.get("preprocessing", {})
    overrides = preprocessing.get("overrides", {})
    if (preprocessing.get("metadata_priority") != "official" or overrides.get("image_size") != [288, 288]
            or overrides.get("resize_size") != 288 or overrides.get("interpolation") != "bicubic"):
        raise EfficientNetB2Error("Configured preprocessing does not match the official EfficientNet-B2 transform.")
    return config


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
