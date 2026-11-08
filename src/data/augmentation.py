"""On-the-fly, split-safe augmentation and Ultralytics configuration adapter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import albumentations as A
import cv2
import numpy as np
from PIL import Image

from src.data.preprocessing import ConfigValidationError, coerce_rgb_image, load_yaml


BORDER_MODES = {"constant": cv2.BORDER_CONSTANT, "reflect_101": cv2.BORDER_REFLECT_101}


def resolve_augmentation_config(profile_name: str, project_root: Path | str | None = None) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    shared = load_yaml(root / "configs" / "augmentation.yaml")
    profile = shared.get("profiles", {}).get(profile_name)
    if not isinstance(profile, dict):
        raise ConfigValidationError(f"Unknown augmentation profile: {profile_name}")
    validate_augmentation_config(profile)
    return dict(profile)


def validate_augmentation_config(config: Mapping[str, Any]) -> None:
    for key, value in config.items():
        if key.endswith("probability") and not 0 <= float(value) <= 1:
            raise ConfigValidationError(f"{key} must be in [0, 1].")
    for key in ("rotation_degrees", "horizontal_shear_degrees", "vertical_shear_degrees", "brightness_limit"):
        if float(config.get(key, 0)) < 0:
            raise ConfigValidationError(f"{key} must be non-negative.")
    if config.get("border_mode", "reflect_101") not in BORDER_MODES:
        raise ConfigValidationError("Unsupported augmentation border_mode.")


@dataclass(frozen=True)
class AlbumentationsImageTransform:
    pipeline: A.Compose
    enabled: bool

    def __call__(self, image: Image.Image | np.ndarray) -> Image.Image:
        rgb = coerce_rgb_image(image)
        if not self.enabled:
            return rgb
        augmented = self.pipeline(image=np.asarray(rgb))["image"]
        return Image.fromarray(augmented).convert("RGB")


def build_training_augmentation(profile_name: str, enabled: bool = True, seed: int = 42, project_root: Path | str | None = None) -> AlbumentationsImageTransform:
    """Build image-domain augmentation. Validation and internal test never call it."""
    config = resolve_augmentation_config(profile_name, project_root)
    if not enabled or not config.get("enabled", False):
        return AlbumentationsImageTransform(A.Compose([]), False)
    border_mode = BORDER_MODES[config["border_mode"]]
    pipeline = A.Compose([
        A.HorizontalFlip(p=float(config["horizontal_flip_probability"])),
        A.Rotate(limit=(-float(config["rotation_degrees"]), float(config["rotation_degrees"])), border_mode=border_mode, p=float(config["rotation_probability"])),
        A.Affine(
            shear={
                "x": (-float(config["horizontal_shear_degrees"]), float(config["horizontal_shear_degrees"])),
                "y": (-float(config["vertical_shear_degrees"]), float(config["vertical_shear_degrees"])),
            },
            border_mode=border_mode,
            p=float(config["shear_probability"]),
        ),
        # Brightness is a bounded approximation, not a physically equivalent exposure simulation.
        A.RandomBrightnessContrast(
            brightness_limit=(-float(config["brightness_limit"]), float(config["brightness_limit"])),
            contrast_limit=(0.0, 0.0),
            p=float(config["brightness_probability"]),
        ),
    ])
    pipeline.set_random_seed(seed)
    return AlbumentationsImageTransform(pipeline, True)


def build_split_augmentation(split: str, profile_name: str, enabled: bool = True, seed: int = 42, project_root: Path | str | None = None) -> AlbumentationsImageTransform | None:
    if split != "train":
        return None
    return build_training_augmentation(profile_name, enabled, seed, project_root)


def build_yolo_classification_adapter(config: Mapping[str, Any] | None = None, project_root: Path | str | None = None) -> dict[str, Any]:
    """Return Ultralytics-owned options without a second torchvision/albumentations pipeline."""
    root = Path(project_root or Path.cwd()).resolve()
    model_config = dict(config) if config is not None else load_yaml(root / "configs" / "models" / "yolov8.yaml")
    yolo = dict(model_config.get("ultralytics", {}))
    if yolo.get("external_augmentation", False):
        raise ConfigValidationError("YOLO adapter forbids external augmentation to prevent double augmentation.")
    if float(yolo.get("flipud", 0.0)) != 0.0:
        raise ConfigValidationError("Vertical flip is not enabled by the baseline YOLO policy.")
    return {
        "imgsz": int(yolo["imgsz"]),
        "fliplr": float(yolo["fliplr"]),
        "flipud": float(yolo["flipud"]),
        "degrees": float(yolo["degrees"]),
        "shear": float(yolo["shear"]),
        "hsv_v": yolo.get("hsv_v"),
        "erasing": float(yolo.get("erasing", 0.0)),
        "external_augmentation": False,
        "normalization_owner": "ultralytics",
        "note": "hsv_v is unset because it is HSV value jitter, not an exposure-equivalent transform.",
    }


def create_augmentation_preview(project_root: Path | str | None = None, seed: int = 42) -> Path:
    """Render a read-only CNN augmentation preview from one active training image."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data.preprocessing import build_transforms

    root = Path(project_root or Path.cwd()).resolve()
    with (root / "data" / "metadata" / "split_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    if not rows:
        raise ConfigValidationError("Active split manifest does not contain training images.")
    row = sorted(rows, key=lambda item: item["source_relative_path"])[0]
    original = coerce_rgb_image(root / row["processed_relative_path"])
    config = resolve_augmentation_config("cnn_standard", root)
    border_mode = BORDER_MODES[config["border_mode"]]
    shear = {"x": (-10.0, 10.0), "y": (-10.0, 10.0)}
    examples = [
        ("Original", original),
        ("Preprocessing 224x224", None),
        ("Horizontal flip", A.Compose([A.HorizontalFlip(p=1.0)])),
        ("Rotation", A.Compose([A.Rotate(limit=(-15, 15), border_mode=border_mode, p=1.0)])),
        ("Shear", A.Compose([A.Affine(shear=shear, border_mode=border_mode, p=1.0)])),
        ("Brightness approximation", A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(0, 0), p=1.0)])),
        ("Combined training augmentation", build_training_augmentation("cnn_standard", True, seed, root)),
    ]
    for index, (_, transform) in enumerate(examples):
        if isinstance(transform, A.Compose):
            transform.set_random_seed(seed + index)
    preprocess = build_transforms("cnn", "val", project_root=root)
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, (title, transform) in zip(axes.flat, examples):
        if title.startswith("Preprocessing"):
            tensor = preprocess(original)
            display = tensor.permute(1, 2, 0).numpy()
        elif isinstance(transform, AlbumentationsImageTransform):
            display = np.asarray(transform(original))
        elif isinstance(transform, A.Compose):
            display = transform(image=np.asarray(original))["image"]
        else:
            display = np.asarray(original)
        axis.imshow(display)
        axis.set_title(f"{row['class_name']}: {title}")
        axis.axis("off")
    axes.flat[-1].axis("off")
    figure.suptitle("TASK 03 augmentation preview (active train image only)")
    figure.tight_layout()
    output = root / "outputs" / "figures" / "augmentation_preview.png"
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def create_augmentation_preview_all_classes(project_root: Path | str | None = None, seed: int = 42) -> Path:
    """Render one original, preprocessed, and augmented active training image per class."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.data.preprocessing import build_transforms

    root = Path(project_root or Path.cwd()).resolve()
    with (root / "data" / "metadata" / "split_manifest.csv").open(encoding="utf-8", newline="") as handle:
        records = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    selected = {}
    for row in sorted(records, key=lambda item: (item["class_name"], item["source_relative_path"])):
        selected.setdefault(row["class_name"], row)
    if len(selected) != 8:
        raise ConfigValidationError(f"Expected eight active classes for preview, found {len(selected)}.")
    preprocess = build_transforms("cnn", "val", project_root=root)
    augmentation = build_training_augmentation("cnn_standard", True, seed, root)
    figure, axes = plt.subplots(len(selected), 3, figsize=(12, 3.2 * len(selected)))
    for row_index, (class_name, row) in enumerate(sorted(selected.items())):
        image = coerce_rgb_image(root / row["processed_relative_path"])
        panels = [("Original", np.asarray(image)), ("Preprocessing", preprocess(image).permute(1, 2, 0).numpy()),
                  ("Training augmentation", np.asarray(augmentation(image)))]
        for axis, (title, pixels) in zip(axes[row_index], panels):
            axis.imshow(pixels)
            axis.set_title(f"{class_name}: {title}")
            axis.axis("off")
    figure.suptitle("TASK 04 augmentation preview: one active training image per class", y=0.998)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    output = root / "outputs" / "figures" / "augmentation_preview_all_classes.png"
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output
