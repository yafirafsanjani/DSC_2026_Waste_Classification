"""Read-only integrity and leakage audit for the extracted Roboflow datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)
VALID_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
EXTENSION_FORMATS = {
    ".jpg": {"JPEG"}, ".jpeg": {"JPEG"}, ".png": {"PNG"}, ".webp": {"WEBP"},
    ".bmp": {"BMP"}, ".tif": {"TIFF"}, ".tiff": {"TIFF"},
}
ROBOFLOW_FILENAME_PATTERN = re.compile(r"^(?P<source>.+?)\.rf\.(?P<roboflow_id>[0-9a-fA-F]+)$")
POPULATION_COUNT = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class ImageRecord:
    """Metadata collected from one readable image without changing the image."""

    dataset: str
    split: str
    class_name: str
    relative_path: str
    filename: str
    extension: str
    byte_size: int
    sha256: str
    perceptual_hash: str
    source_identity: str | None
    roboflow_id: str | None
    image_format: str
    width: int
    height: int
    mode: str

    @property
    def context(self) -> str:
        return f"{self.dataset}/{self.split}"


@dataclass(frozen=True)
class AuditConfig:
    """Configurable locations and threshold for a deterministic audit run."""

    project_root: Path
    dataset_versions: tuple[str, ...] = ("v1", "v5")
    near_duplicate_threshold: int = 8

    @property
    def extracted_root(self) -> Path:
        return self.project_root / "data" / "extracted"

    @property
    def metadata_dir(self) -> Path:
        return self.project_root / "data" / "metadata"

    @property
    def figures_dir(self) -> Path:
        return self.project_root / "outputs" / "figures"


def extract_roboflow_source_identity(filename: str) -> tuple[str | None, str | None]:
    """Return pre-``.rf`` source name and Roboflow identifier when present."""

    match = ROBOFLOW_FILENAME_PATTERN.match(Path(filename).stem)
    if match is None:
        return None, None
    return match.group("source"), match.group("roboflow_id")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a SHA-256 digest using bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image: Image.Image) -> str:
    """Calculate a 64-bit DCT perceptual hash from a decoded image."""

    size, hash_size = 32, 8
    grayscale = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float32)
    coordinates = np.arange(size, dtype=np.float32)
    frequencies = np.arange(size, dtype=np.float32)[:, None]
    dct_basis = np.cos(np.pi * (coordinates + 0.5) * frequencies / size)
    dct_basis[0] *= np.sqrt(1.0 / size)
    dct_basis[1:] *= np.sqrt(2.0 / size)
    coefficients = dct_basis @ pixels @ dct_basis.T
    low_frequency = coefficients[:hash_size, :hash_size].reshape(-1)
    median = np.median(low_frequency[1:])
    bits = low_frequency > median
    bits[0] = False
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_distance(left_hash: str, right_hash: str) -> int:
    """Return the bit distance between two hexadecimal perceptual hashes."""

    return (int(left_hash, 16) ^ int(right_hash, 16)).bit_count()


def _relative_path(path: Path, project_root: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _inspect_image(
    path: Path, dataset: str, split: str, class_name: str, project_root: Path
) -> tuple[ImageRecord | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Validate and collect one image; errors are returned as data, not raised."""

    relative_path = _relative_path(path, project_root)
    try:
        with Image.open(path) as opened_image:
            opened_image.verify()
        with Image.open(path) as opened_image:
            opened_image.load()
            image_format = opened_image.format or "UNKNOWN"
            width, height = opened_image.size
            image_mode = opened_image.mode
            image_hash = perceptual_hash(opened_image)
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return None, {
            "relative_path": relative_path, "dataset": dataset, "split": split,
            "class_name": class_name, "issue_type": "unreadable_image", "error": str(error),
        }, None

    extension = path.suffix.lower()
    format_mismatch = None
    if image_format not in EXTENSION_FORMATS[extension]:
        format_mismatch = {
            "relative_path": relative_path, "dataset": dataset, "split": split,
            "class_name": class_name, "issue_type": "extension_format_mismatch",
            "extension": extension, "detected_format": image_format,
        }
    source_identity, roboflow_id = extract_roboflow_source_identity(path.name)
    return ImageRecord(
        dataset=dataset, split=split, class_name=class_name, relative_path=relative_path,
        filename=path.name, extension=extension, byte_size=path.stat().st_size,
        sha256=sha256_file(path), perceptual_hash=image_hash,
        source_identity=source_identity, roboflow_id=roboflow_id, image_format=image_format,
        width=width, height=height, mode=image_mode,
    ), None, format_mismatch


def scan_datasets(config: AuditConfig) -> tuple[list[ImageRecord], list[dict[str, Any]], dict[str, Any]]:
    """Scan configured versions and return readable images, issues, and counts."""

    records: list[ImageRecord] = []
    issues: list[dict[str, Any]] = []
    structure: dict[str, Any] = {}
    for dataset in config.dataset_versions:
        dataset_root = config.extracted_root / dataset
        dataset_summary: dict[str, Any] = {"splits": {}, "classes": []}
        structure[dataset] = dataset_summary
        if not dataset_root.is_dir():
            issues.append({
                "relative_path": _relative_path(dataset_root, config.project_root), "dataset": dataset,
                "split": None, "class_name": None, "issue_type": "missing_dataset_directory",
                "error": "Dataset directory does not exist.",
            })
            continue
        class_names: set[str] = set()
        for split_directory in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
            split = split_directory.name
            class_directories = sorted(path for path in split_directory.iterdir() if path.is_dir())
            split_summary = {
                "candidate_image_files": 0, "readable_images": 0,
                "class_counts": {}, "class_candidate_counts": {},
            }
            dataset_summary["splits"][split] = split_summary
            for class_directory in class_directories:
                class_name = class_directory.name
                class_names.add(class_name)
                candidate_count = 0
                readable_count = 0
                for path in sorted(item for item in class_directory.rglob("*") if item.is_file()):
                    extension = path.suffix.lower()
                    if extension not in VALID_IMAGE_EXTENSIONS:
                        issues.append({
                            "relative_path": _relative_path(path, config.project_root), "dataset": dataset,
                            "split": split, "class_name": class_name,
                            "issue_type": "invalid_image_extension", "extension": extension or None,
                            "error": "Unsupported extension in a class directory.",
                        })
                        continue
                    candidate_count += 1
                    record, image_issue, format_mismatch = _inspect_image(
                        path, dataset, split, class_name, config.project_root
                    )
                    if image_issue is not None:
                        issues.append(image_issue)
                        continue
                    if format_mismatch is not None:
                        issues.append(format_mismatch)
                    if record is not None:
                        records.append(record)
                        readable_count += 1
                split_summary["class_candidate_counts"][class_name] = candidate_count
                split_summary["class_counts"][class_name] = readable_count
                split_summary["candidate_image_files"] += candidate_count
                split_summary["readable_images"] += readable_count
        dataset_summary["classes"] = sorted(class_names)
    return records, issues, structure


def group_exact_duplicates(records: Iterable[ImageRecord]) -> list[dict[str, Any]]:
    """Group byte-identical readable images by SHA-256."""

    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)
    groups: list[dict[str, Any]] = []
    for digest, group in sorted(by_hash.items()):
        if len(group) >= 2:
            groups.append({
                "duplicate_type": "exact_sha256", "sha256": digest, "image_count": len(group),
                "contexts": sorted({record.context for record in group}),
                "classes": sorted({record.class_name for record in group}),
                "paths": [record.relative_path for record in sorted(group, key=lambda item: item.relative_path)],
            })
    return groups


def find_near_duplicates(records: Sequence[ImageRecord], threshold: int) -> list[dict[str, Any]]:
    """Find visual candidates with DCT perceptual-hash distance at most ``threshold``."""

    if threshold < 0:
        raise ValueError("near_duplicate_threshold must be non-negative.")
    if len(records) < 2:
        return []
    hash_values = np.asarray([int(record.perceptual_hash, 16) for record in records], dtype=np.uint64)
    pairs: list[dict[str, Any]] = []
    for left_index, left_record in enumerate(records[:-1]):
        xor_values = hash_values[left_index] ^ hash_values[left_index + 1:]
        distances = POPULATION_COUNT[xor_values.view(np.uint8)].reshape(-1, 8).sum(axis=1, dtype=np.uint8)
        for offset in np.flatnonzero(distances <= threshold):
            right_index = left_index + 1 + int(offset)
            right_record = records[right_index]
            if left_record.sha256 == right_record.sha256:
                continue
            pairs.append({
                "duplicate_type": "perceptual_hash", "hamming_distance": int(distances[int(offset)]),
                "left_path": left_record.relative_path, "right_path": right_record.relative_path,
                "left_context": left_record.context, "right_context": right_record.context,
                "left_class": left_record.class_name, "right_class": right_record.class_name,
                "same_source_identity": (
                    left_record.source_identity is not None
                    and left_record.source_identity == right_record.source_identity
                ),
            })
    return pairs


def group_source_identity(records: Iterable[ImageRecord]) -> list[dict[str, Any]]:
    """Group Roboflow-derived source identities, including augmented variants."""

    by_source: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.source_identity is not None:
            by_source[record.source_identity].append(record)
    groups: list[dict[str, Any]] = []
    for source_identity, group in sorted(by_source.items()):
        if len(group) >= 2:
            groups.append({
                "duplicate_type": "roboflow_source_identity", "source_identity": source_identity,
                "image_count": len(group), "contexts": sorted({record.context for record in group}),
                "classes": sorted({record.class_name for record in group}),
                "paths": [record.relative_path for record in sorted(group, key=lambda item: item.relative_path)],
            })
    return groups


def build_leakage_report(
    exact_groups: Iterable[dict[str, Any]], source_groups: Iterable[dict[str, Any]],
    near_pairs: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create cross-context leakage evidence from independent audit signals."""

    leakage_rows: list[dict[str, Any]] = []
    for group in exact_groups:
        if len(group["contexts"]) > 1:
            leakage_rows.append({
                "evidence_type": "exact_sha256", "severity": "high", "contexts": group["contexts"],
                "classes": group["classes"], "paths": group["paths"],
                "details": f"SHA-256: {group['sha256']}",
            })
    for group in source_groups:
        if len(group["contexts"]) > 1:
            leakage_rows.append({
                "evidence_type": "roboflow_source_identity", "severity": "high",
                "contexts": group["contexts"], "classes": group["classes"], "paths": group["paths"],
                "details": f"Roboflow source identity: {group['source_identity']}",
            })
    for pair in near_pairs:
        if pair["left_context"] != pair["right_context"]:
            leakage_rows.append({
                "evidence_type": "perceptual_hash",
                "severity": "high" if pair["same_source_identity"] else "review",
                "contexts": sorted({pair["left_context"], pair["right_context"]}),
                "classes": sorted({pair["left_class"], pair["right_class"]}),
                "paths": [pair["left_path"], pair["right_path"]],
                "details": f"Perceptual-hash Hamming distance: {pair['hamming_distance']}",
            })
    return leakage_rows


def _serialize_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write rows as UTF-8 CSV, preserving a header for empty reports."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames, rows = ["status"], [{"status": "no_records"}]
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize_cell(row.get(key)) for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a human-readable JSON audit summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2, sort_keys=True)
        file_handle.write("\n")


def plot_class_distribution(structure: dict[str, Any], output_path: Path) -> None:
    """Save a grouped bar chart of readable images by dataset and split."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[tuple[str, str, dict[str, int]]] = []
    class_names: set[str] = set()
    for dataset, dataset_summary in structure.items():
        for split, split_summary in dataset_summary["splits"].items():
            counts = split_summary["class_counts"]
            rows.append((dataset, split, counts))
            class_names.update(counts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(15, 7))
    if not rows or not class_names:
        axis.text(0.5, 0.5, "No readable images found", ha="center", va="center")
        axis.set_axis_off()
    else:
        sorted_classes = sorted(class_names)
        x_positions = np.arange(len(sorted_classes), dtype=np.float64)
        width = 0.8 / len(rows)
        for index, (dataset, split, counts) in enumerate(rows):
            offset = (index - (len(rows) - 1) / 2) * width
            axis.bar(
                x_positions + offset,
                [counts.get(class_name, 0) for class_name in sorted_classes],
                width=width,
                label=f"{dataset}/{split}",
            )
        axis.set_xticks(x_positions, sorted_classes, rotation=30, ha="right")
        axis.set_ylabel("Readable image count")
        axis.set_title("Waste Dataset Class Distribution by Export and Split")
        axis.legend(ncol=2, fontsize=9)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _split_policies() -> dict[str, str]:
    return {
        "v1/train": "candidate_for_future_training_only",
        "v1/test": "excluded_from_training_validation_evaluation_and_model_selection; audit_only",
        "v5/train": "audit_only; do_not_use_for_training_or_splitting_yet",
        "v5/valid": "audit_only; do_not_use_for_training_or_splitting_yet",
        "v5/test": "excluded_from_training_validation_evaluation_and_model_selection; audit_only",
    }


def run_audit(
    project_root: Path | str, near_duplicate_threshold: int = 8,
    dataset_versions: tuple[str, ...] = ("v1", "v5"),
) -> dict[str, Any]:
    """Run the full read-only audit and return the saved JSON summary data."""

    root = Path(project_root).resolve()
    config = AuditConfig(root, dataset_versions, near_duplicate_threshold)
    LOGGER.info("Scanning dataset versions: %s", ", ".join(config.dataset_versions))
    records, issues, structure = scan_datasets(config)
    LOGGER.info("Collected %d readable images and %d integrity issues.", len(records), len(issues))
    exact_groups = group_exact_duplicates(records)
    source_groups = group_source_identity(records)
    near_pairs = find_near_duplicates(records, config.near_duplicate_threshold)
    leakage_rows = build_leakage_report(exact_groups, source_groups, near_pairs)
    version_classes = {dataset: set(summary["classes"]) for dataset, summary in structure.items()}
    v1_classes = version_classes.get("v1", set())
    v5_classes = version_classes.get("v5", set())
    issue_counts = Counter(issue["issue_type"] for issue in issues)
    evidence_counts = Counter(row["evidence_type"] for row in leakage_rows)
    totals = {
        "readable_images": len(records),
        "candidate_image_files": sum(
            split["candidate_image_files"]
            for dataset in structure.values()
            for split in dataset["splits"].values()
        ),
        "integrity_issues": len(issues),
        "exact_duplicate_groups": len(exact_groups),
        "near_duplicate_pairs": len(near_pairs),
        "roboflow_source_identity_groups": len(source_groups),
        "cross_context_leakage_findings": len(leakage_rows),
    }
    summary: dict[str, Any] = {
        "audit_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "project_root": str(root),
        "dataset_root": _relative_path(config.extracted_root, root),
        "dataset_versions": list(config.dataset_versions),
        "near_duplicate_threshold": config.near_duplicate_threshold,
        "split_policies": _split_policies(),
        "structure": structure,
        "class_comparison": {
            "v1_classes": sorted(v1_classes), "v5_classes": sorted(v5_classes),
            "common_classes": sorted(v1_classes & v5_classes), "only_v1": sorted(v1_classes - v5_classes),
            "only_v5": sorted(v5_classes - v1_classes),
        },
        "totals": totals,
        "integrity_issue_counts": dict(sorted(issue_counts.items())),
        "leakage_evidence_counts": dict(sorted(evidence_counts.items())),
        "notes": [
            "Test splits are read only for integrity and leakage detection, never for model selection.",
            "A shared Roboflow source identity is evidence of a common source even when file SHA-256 differs.",
            "Perceptual-hash candidates should be reviewed visually before deleting or excluding any image.",
        ],
    }
    manifest_rows = [asdict(record) for record in records]
    distribution_rows = [
        {
            "dataset": dataset, "split": split, "class_name": class_name, "readable_images": count,
            "candidate_image_files": dataset_summary["splits"][split]["class_candidate_counts"].get(class_name, 0),
        }
        for dataset, dataset_summary in structure.items()
        for split, split_summary in dataset_summary["splits"].items()
        for class_name, count in split_summary["class_counts"].items()
    ]
    write_csv(config.metadata_dir / "dataset_audit_image_manifest.csv", manifest_rows)
    write_csv(config.metadata_dir / "dataset_audit_distribution.csv", distribution_rows)
    write_csv(config.metadata_dir / "dataset_audit_integrity_issues.csv", issues)
    write_csv(config.metadata_dir / "dataset_audit_exact_duplicates.csv", exact_groups)
    write_csv(config.metadata_dir / "dataset_audit_near_duplicates.csv", near_pairs)
    write_csv(config.metadata_dir / "dataset_audit_source_identity_groups.csv", source_groups)
    write_csv(config.metadata_dir / "dataset_audit_leakage.csv", leakage_rows)
    write_json(config.metadata_dir / "dataset_audit_summary.json", summary)
    plot_class_distribution(structure, config.figures_dir / "dataset_class_distribution.png")
    LOGGER.info("Audit complete: %d exact groups, %d near pairs, %d leakage findings.",
                len(exact_groups), len(near_pairs), len(leakage_rows))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a read-only dataset integrity audit.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                        help="Project root containing data/extracted (default: current directory).")
    parser.add_argument("--near-duplicate-threshold", type=int, default=8,
                        help="Maximum 64-bit perceptual-hash Hamming distance (default: 8).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python -m src.data.data_audit``."""

    arguments = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_audit(arguments.project_root, arguments.near_duplicate_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
