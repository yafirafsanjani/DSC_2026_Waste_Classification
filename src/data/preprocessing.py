"""Configurable, model-aware, read-only image preprocessing transforms."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.transforms import InterpolationMode, functional as TF


class ConfigValidationError(ValueError):
    """Raised when a transform profile is unsafe or incomplete."""


class InvalidImageError(ValueError):
    """Raised when an image cannot be decoded or converted to RGB."""


INTERPOLATIONS = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
}


def load_yaml(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigValidationError(f"YAML root must be a mapping: {path}")
    return data


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _project_root(project_root: Path | str | None) -> Path:
    return Path(project_root or Path.cwd()).resolve()


def resolve_model_config(model_type: str, config: Mapping[str, Any] | None = None, project_root: Path | str | None = None) -> dict[str, Any]:
    """Resolve a model profile without mutating shared YAML dictionaries."""
    root = _project_root(project_root)
    shared = load_yaml(root / "configs" / "preprocessing.yaml")
    model_config = dict(config) if config is not None else load_yaml(root / "configs" / "models" / f"{model_type}.yaml")
    preprocessing = model_config.get("preprocessing", {})
    profile_name = preprocessing.get("profile")
    profile = shared.get("profiles", {}).get(profile_name)
    if not isinstance(profile, dict):
        raise ConfigValidationError(f"Unknown preprocessing profile: {profile_name}")
    resolved = _merge(profile, preprocessing.get("overrides", {}))
    resolved["model_type"] = model_config.get("model_type", model_type)
    resolved["framework"] = model_config.get("framework", "custom_pytorch")
    resolved["pretrained"] = bool(model_config.get("pretrained", False))
    resolved["metadata_priority"] = preprocessing.get("metadata_priority", shared.get("metadata_priority", "official"))
    resolved["weights_name"] = model_config.get("weights")
    resolved["augmentation"] = model_config.get("augmentation", {})
    validate_preprocessing_config(resolved)
    return resolved


def validate_preprocessing_config(config: Mapping[str, Any]) -> None:
    size = config.get("image_size")
    if not (isinstance(size, (list, tuple)) and len(size) == 2 and all(isinstance(x, int) and x > 0 for x in size)):
        raise ConfigValidationError("image_size must contain two positive integers.")
    if config.get("color_mode") != "RGB":
        raise ConfigValidationError("Only RGB color mode is supported.")
    if config.get("resize_mode") not in {"direct", "pad", "resize_center_crop"}:
        raise ConfigValidationError("resize_mode must be direct, pad, or resize_center_crop.")
    if config.get("interpolation") not in INTERPOLATIONS:
        raise ConfigValidationError("Unknown interpolation method.")
    scaling = config.get("pixel_scaling")
    if scaling not in {"zero_one", "none", "pipeline_owned"}:
        raise ConfigValidationError("Unsupported pixel_scaling value.")
    normalization = config.get("normalization")
    if normalization is not None:
        mean, std = normalization.get("mean"), normalization.get("std")
        if not (isinstance(mean, list) and isinstance(std, list) and len(mean) == len(std) == 3):
            raise ConfigValidationError("normalization mean/std must each have three values.")
        if any(float(value) == 0 for value in std):
            raise ConfigValidationError("normalization std cannot contain zero.")
        if scaling != "zero_one":
            raise ConfigValidationError("normalization requires zero_one pixel scaling.")


def coerce_rgb_image(image: Image.Image | np.ndarray | Path | str) -> Image.Image:
    """Decode supported input safely and convert it to an independent RGB image."""
    try:
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                opened.load()
                return ImageOps.exif_transpose(opened).convert("RGB")
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim not in {2, 3}:
                raise InvalidImageError("NumPy image must have two or three dimensions.")
            return Image.fromarray(image).convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise InvalidImageError(f"Unable to decode image as RGB: {error}") from error
    raise InvalidImageError(f"Unsupported image type: {type(image).__name__}")


def _resize(image: Image.Image, config: Mapping[str, Any]) -> Image.Image:
    height, width = config["image_size"]
    interpolation = INTERPOLATIONS[config["interpolation"]]
    mode = config["resize_mode"]
    if mode == "direct":
        return TF.resize(image, [height, width], interpolation=interpolation)
    if mode == "pad":
        contained = ImageOps.contain(image, (width, height), method=Image.Resampling.BICUBIC if interpolation == InterpolationMode.BICUBIC else Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        canvas.paste(contained, ((width - contained.width) // 2, (height - contained.height) // 2))
        return canvas
    resize_size = config.get("resize_size", max(height, width))
    resized = TF.resize(image, resize_size, interpolation=interpolation)
    return TF.center_crop(resized, [height, width])


def validate_tensor(tensor: torch.Tensor, expected_size: tuple[int, int], normalized: bool) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.shape != (3, *expected_size):
        raise InvalidImageError(f"Expected CHW tensor (3, {expected_size[0]}, {expected_size[1]}), got {getattr(tensor, 'shape', None)}")
    if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
        raise InvalidImageError("Tensor must be finite float32.")
    if not normalized and (tensor.min() < 0 or tensor.max() > 1):
        raise InvalidImageError("Unnormalized zero_one tensor must remain in [0, 1].")
    return tensor


@dataclass(frozen=True)
class ModelAwareTransform:
    config: Mapping[str, Any]
    preprocess: Callable[[Image.Image], torch.Tensor]
    augmentation: Callable[[Image.Image], Image.Image] | None = None
    metadata_source: str = "config"

    def __call__(self, image: Image.Image | np.ndarray | Path | str) -> torch.Tensor:
        rgb = coerce_rgb_image(image)
        if self.augmentation is not None:
            rgb = self.augmentation(rgb)
        tensor = self.preprocess(rgb)
        return validate_tensor(tensor, tuple(self.config["image_size"]), self.metadata_source != "config" or self.config.get("normalization") is not None)


def _config_preprocess(config: Mapping[str, Any]) -> Callable[[Image.Image], torch.Tensor]:
    def apply(image: Image.Image) -> torch.Tensor:
        resized = _resize(image, config)
        scaling = config["pixel_scaling"]
        if scaling == "none":
            tensor = TF.pil_to_tensor(resized).to(torch.float32)
        else:
            tensor = TF.to_tensor(resized)
        normalization = config.get("normalization")
        if normalization is not None:
            tensor = TF.normalize(tensor, normalization["mean"], normalization["std"])
        return tensor
    return apply


def _official_torchvision_preprocess(weights: Any) -> Callable[[Image.Image], torch.Tensor]:
    transforms = weights.transforms()
    return lambda image: transforms(image)


def _official_timm_preprocess(model: Any) -> Callable[[Image.Image], torch.Tensor]:
    from timm.data import create_transform, resolve_model_data_config
    data_config = resolve_model_data_config(model=model)
    transform = create_transform(**data_config, is_training=False)
    return lambda image: transform(image)


def build_transforms(model_type: str, split: str, model: Any = None, weights: Any = None, config: Mapping[str, Any] | None = None, project_root: Path | str | None = None, augmentation: Callable[[Image.Image], Image.Image] | None = None) -> ModelAwareTransform:
    """Build split-safe transform; official pretrained metadata wins when supplied."""
    if split not in {"train", "val", "test_internal", "inference"}:
        raise ConfigValidationError(f"Unsupported split: {split}")
    resolved = resolve_model_config(model_type, config, project_root)
    metadata_source = "config"
    if resolved["metadata_priority"] == "official" and weights is not None:
        preprocess = _official_torchvision_preprocess(weights)
        metadata_source = "torchvision_weights"
    elif resolved["metadata_priority"] == "official" and resolved["framework"] == "timm" and model is not None:
        preprocess = _official_timm_preprocess(model)
        metadata_source = "timm_model_data_config"
    else:
        preprocess = _config_preprocess(resolved)
    # Random image-level augmentation is allowed only before tensor preprocessing on train.
    active_augmentation = augmentation if split == "train" else None
    if split != "train" and augmentation is not None:
        active_augmentation = None
    return ModelAwareTransform(resolved, preprocess, active_augmentation, metadata_source)


def load_active_class_mapping(project_root: Path | str | None = None) -> dict[str, int]:
    root = _project_root(project_root)
    with (root / "data" / "metadata" / "split_manifest.csv").open(encoding="utf-8", newline="") as handle:
        classes = sorted({row["class_name"] for row in __import__("csv").DictReader(handle)})
    if not classes:
        raise ConfigValidationError("Active split manifest has no classes.")
    return {class_name: index for index, class_name in enumerate(classes)}
