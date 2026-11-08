from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights

from src.data.augmentation import (
    build_split_augmentation,
    build_training_augmentation,
    build_yolo_classification_adapter,
    resolve_augmentation_config,
)
from src.data.preprocessing import (
    ConfigValidationError,
    InvalidImageError,
    build_transforms,
    coerce_rgb_image,
    resolve_model_config,
)


def _image(mode: str = "RGB") -> Image.Image:
    image = Image.new(mode, (83, 51), (20, 120, 220, 100) if mode == "RGBA" else (20, 120, 220))
    return image


def test_rgb_resize_tensor_dtype_and_range() -> None:
    transform = build_transforms("cnn", "val")
    tensor = transform(_image("RGBA"))
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert 0 <= tensor.min() <= tensor.max() <= 1
    assert coerce_rgb_image(_image("RGBA")).mode == "RGB"


def test_normalization_and_no_double_normalization() -> None:
    config = {
        "model_type": "cnn",
        "framework": "custom_pytorch",
        "preprocessing": {
            "profile": "cnn_default",
            "overrides": {"normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
        },
    }
    tensor = build_transforms("cnn", "val", config=config)(_image())
    assert tensor.shape == (3, 224, 224)
    assert torch.isfinite(tensor).all()
    assert tensor.min() >= -1 and tensor.max() <= 1
    with pytest.raises(ConfigValidationError):
        build_transforms("cnn", "val", config={
            "preprocessing": {"profile": "cnn_default", "overrides": {"pixel_scaling": "none", "normalization": {"mean": [0, 0, 0], "std": [1, 1, 1]}}},
        })


def test_validation_and_internal_test_are_deterministic() -> None:
    image = _image()
    for split in ("val", "test_internal"):
        transform = build_transforms("cnn", split, augmentation=build_training_augmentation("cnn_standard", seed=42))
        assert torch.equal(transform(image), transform(image))


def test_training_augmentation_can_be_enabled_or_disabled() -> None:
    image = _image()
    enabled = build_training_augmentation("cnn_standard", enabled=True, seed=42)
    disabled = build_training_augmentation("cnn_standard", enabled=False, seed=42)
    assert enabled.enabled
    assert not disabled.enabled
    assert disabled(image).tobytes() == coerce_rgb_image(image).tobytes()
    assert build_split_augmentation("val", "cnn_standard") is None


def test_model_profiles_are_independent_and_metadata_transform_is_supported() -> None:
    cnn = resolve_model_config("cnn")
    mobile = resolve_model_config("mobilenetv3")
    resnet = resolve_model_config("resnet50")
    efficientnet = resolve_model_config("efficientnet")
    assert cnn["normalization"] is None
    assert mobile["normalization"] == resnet["normalization"]
    assert efficientnet["interpolation"] == "bicubic"
    transform = build_transforms("mobilenetv3", "val", weights=MobileNet_V3_Large_Weights.DEFAULT)
    tensor = transform(_image())
    assert transform.metadata_source == "torchvision_weights"
    assert tensor.shape == (3, 224, 224)
    assert torch.isfinite(tensor).all()


def test_augmentation_yaml_and_yolo_adapter_prevent_double_augmentation() -> None:
    profile = resolve_augmentation_config("cnn_standard")
    assert profile["rotation_degrees"] == 15
    assert profile["horizontal_shear_degrees"] == 10
    assert profile["vertical_shear_degrees"] == 10
    adapter = build_yolo_classification_adapter()
    assert adapter["external_augmentation"] is False
    assert adapter["normalization_owner"] == "ultralytics"
    assert adapter["flipud"] == 0.0


def test_invalid_image_and_manifest_are_not_modified(tmp_path: Path) -> None:
    invalid = tmp_path / "broken.jpg"
    invalid.write_bytes(b"not an image")
    with pytest.raises(InvalidImageError, match="Unable to decode image"):
        build_transforms("cnn", "val")(invalid)
    manifest = Path("data/metadata/split_manifest.csv")
    before = manifest.read_bytes()
    source = tmp_path / "source.png"
    _image().save(source)
    source_before = source.read_bytes()
    _ = build_transforms("cnn", "train", augmentation=build_training_augmentation("cnn_standard", seed=42))(source)
    assert manifest.read_bytes() == before
    assert source.read_bytes() == source_before
