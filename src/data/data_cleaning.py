"""Deterministic, read-only cleaning for the permitted V1 training images."""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, UnidentifiedImageError

from src.data.data_audit import (
    VALID_IMAGE_EXTENSIONS,
    extract_roboflow_source_identity,
    find_near_duplicates,
    perceptual_hash,
    sha256_file,
    write_csv,
)

LOGGER = logging.getLogger(__name__)


class CriticalDataConflictError(RuntimeError):
    """Raised when labels cannot be retained safely without a human decision."""


@dataclass(frozen=True)
class CleanedImageRecord:
    """One valid V1/train record and its deterministic cleaning decision."""

    source_path: Path
    source_relative_path: str
    class_name: str
    filename: str
    sha256: str
    perceptual_hash: str
    source_identity: str | None
    roboflow_id: str | None
    byte_size: int
    width: int
    height: int
    status: str
    decision_reason: str

    @property
    def relative_path(self) -> str:
        """Compatibility alias for duplicate-detection utilities."""

        return self.source_relative_path

    @property
    def context(self) -> str:
        return "v1/train"


@dataclass(frozen=True)
class CleaningResult:
    """The reportable result of one V1/train-only cleaning run."""

    project_root: Path
    source_root: Path
    retained_records: tuple[CleanedImageRecord, ...]
    manifest_records: tuple[CleanedImageRecord, ...]
    report_rows: tuple[dict[str, Any], ...]
    critical_conflicts: tuple[dict[str, Any], ...]
    near_duplicate_pairs: tuple[dict[str, Any], ...]

    @property
    def is_safe_to_split(self) -> bool:
        return not self.critical_conflicts


def get_v1_train_root(project_root: Path | str) -> Path:
    """Return the sole allowed image source for TASK 02."""

    return Path(project_root).resolve() / "data" / "extracted" / "v1" / "train"


def assert_v1_train_source(path: Path, project_root: Path | str) -> None:
    """Reject any source path outside the explicit V1/train boundary."""

    source_root = get_v1_train_root(project_root)
    resolved_path = path.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Permitted source directory does not exist: {source_root}")
    if not resolved_path.is_relative_to(source_root.resolve()):
        raise ValueError(f"TASK 02 only accepts sources below {source_root}: {resolved_path}")


def _relative(path: Path, project_root: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _read_valid_image(path: Path, project_root: Path, class_name: str) -> CleanedImageRecord:
    """Decode one permitted image and calculate stable duplicate identifiers."""

    assert_v1_train_source(path, project_root)
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            image_hash = perceptual_hash(image)
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ValueError(f"Unreadable V1/train image: {path} ({error})") from error

    source_identity, roboflow_id = extract_roboflow_source_identity(path.name)
    return CleanedImageRecord(
        source_path=path.resolve(),
        source_relative_path=_relative(path, project_root),
        class_name=class_name,
        filename=path.name,
        sha256=sha256_file(path),
        perceptual_hash=image_hash,
        source_identity=source_identity,
        roboflow_id=roboflow_id,
        byte_size=path.stat().st_size,
        width=width,
        height=height,
        status="pending",
        decision_reason="pending_cleaning_decision",
    )


def scan_v1_train(project_root: Path | str) -> tuple[list[CleanedImageRecord], list[dict[str, Any]]]:
    """Scan only V1/train folders and collect integrity issues as report rows."""

    root = Path(project_root).resolve()
    source_root = get_v1_train_root(root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Permitted source directory does not exist: {source_root}")

    records: list[CleanedImageRecord] = []
    report_rows: list[dict[str, Any]] = []
    for class_directory in sorted(path for path in source_root.iterdir() if path.is_dir()):
        for path in sorted(item for item in class_directory.rglob("*") if item.is_file()):
            if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                report_rows.append({
                    "record_type": "integrity_issue",
                    "issue_type": "invalid_image_extension",
                    "source_relative_path": _relative(path, root),
                    "class_name": class_directory.name,
                    "decision": "excluded",
                    "details": f"Unsupported extension: {path.suffix or '<none>'}",
                })
                continue
            try:
                records.append(_read_valid_image(path, root, class_directory.name))
            except ValueError as error:
                report_rows.append({
                    "record_type": "integrity_issue",
                    "issue_type": "unreadable_image",
                    "source_relative_path": _relative(path, root),
                    "class_name": class_directory.name,
                    "decision": "excluded",
                    "details": str(error),
                })
    return records, report_rows


def _with_decision(record: CleanedImageRecord, status: str, reason: str) -> CleanedImageRecord:
    return CleanedImageRecord(
        source_path=record.source_path,
        source_relative_path=record.source_relative_path,
        class_name=record.class_name,
        filename=record.filename,
        sha256=record.sha256,
        perceptual_hash=record.perceptual_hash,
        source_identity=record.source_identity,
        roboflow_id=record.roboflow_id,
        byte_size=record.byte_size,
        width=record.width,
        height=record.height,
        status=status,
        decision_reason=reason,
    )


def resolve_exact_duplicates(
    records: Iterable[CleanedImageRecord],
) -> tuple[list[CleanedImageRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep one lexicographic representative per same-label exact duplicate group."""

    by_hash: dict[str, list[CleanedImageRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)

    resolved: list[CleanedImageRecord] = []
    report_rows: list[dict[str, Any]] = []
    critical_conflicts: list[dict[str, Any]] = []
    for digest, group in sorted(by_hash.items()):
        ordered_group = sorted(group, key=lambda item: item.source_relative_path)
        class_names = sorted({record.class_name for record in ordered_group})
        if len(ordered_group) == 1:
            resolved.append(_with_decision(ordered_group[0], "kept", "unique_sha256"))
            continue
        if len(class_names) > 1:
            conflict = {
                "record_type": "critical_conflict",
                "issue_type": "exact_duplicate_label_conflict",
                "sha256": digest,
                "class_names": class_names,
                "source_paths": [record.source_relative_path for record in ordered_group],
                "decision": "manual_review_required",
                "details": "Byte-identical images have different class labels.",
            }
            critical_conflicts.append(conflict)
            report_rows.append(conflict)
            resolved.extend(
                _with_decision(record, "conflict", "exact_duplicate_label_conflict")
                for record in ordered_group
            )
            continue

        representative = ordered_group[0]
        resolved.append(_with_decision(representative, "kept", "exact_duplicate_representative"))
        duplicate_paths = [record.source_relative_path for record in ordered_group]
        report_rows.append({
            "record_type": "duplicate_decision",
            "issue_type": "exact_duplicate_same_class",
            "sha256": digest,
            "class_names": class_names,
            "source_paths": duplicate_paths,
            "representative_path": representative.source_relative_path,
            "decision": "kept_one_lexicographic_representative",
            "details": f"Excluded {len(ordered_group) - 1} same-class byte-identical image(s).",
        })
        resolved.extend(
            _with_decision(record, "excluded_exact_duplicate", "same_class_exact_duplicate")
            for record in ordered_group[1:]
        )
    return resolved, report_rows, critical_conflicts


def detect_source_label_conflicts(records: Iterable[CleanedImageRecord]) -> list[dict[str, Any]]:
    """Flag a source identity that is associated with more than one class label."""

    by_source: dict[str, list[CleanedImageRecord]] = defaultdict(list)
    for record in records:
        if record.status == "kept" and record.source_identity is not None:
            by_source[record.source_identity].append(record)

    conflicts: list[dict[str, Any]] = []
    for source_identity, group in sorted(by_source.items()):
        class_names = sorted({record.class_name for record in group})
        if len(class_names) > 1:
            conflicts.append({
                "record_type": "critical_conflict",
                "issue_type": "source_identity_label_conflict",
                "source_identity": source_identity,
                "class_names": class_names,
                "source_paths": [record.source_relative_path for record in group],
                "decision": "manual_review_required",
                "details": "One Roboflow source identity has multiple class labels.",
            })
    return conflicts


def build_cleaning_report(
    scanned_records: Sequence[CleanedImageRecord],
    manifest_records: Sequence[CleanedImageRecord],
    integrity_rows: Sequence[dict[str, Any]],
    duplicate_rows: Sequence[dict[str, Any]],
    source_conflicts: Sequence[dict[str, Any]],
    near_pairs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a compact report while retaining near duplicates for manual review."""

    status_counts = Counter(record.status for record in manifest_records)
    report_rows: list[dict[str, Any]] = [
        {"record_type": "summary", "metric": "scanned_image_files", "value": len(scanned_records)},
        {"record_type": "summary", "metric": "retained_images", "value": status_counts["kept"]},
        {"record_type": "summary", "metric": "excluded_exact_duplicates", "value": status_counts["excluded_exact_duplicate"]},
        {"record_type": "summary", "metric": "critical_conflicts", "value": len(source_conflicts) + sum(1 for row in duplicate_rows if row.get("record_type") == "critical_conflict")},
        {"record_type": "summary", "metric": "near_duplicate_pairs", "value": len(near_pairs)},
        *integrity_rows,
        *duplicate_rows,
        *source_conflicts,
    ]
    for pair in near_pairs:
        report_rows.append({
            "record_type": "near_duplicate",
            "issue_type": "perceptual_hash_candidate",
            "left_path": pair["left_path"],
            "right_path": pair["right_path"],
            "hamming_distance": pair["hamming_distance"],
            "same_source_identity": pair["same_source_identity"],
            "decision": "retained_for_manual_review",
            "details": "Near duplicates are never removed automatically.",
        })
    return report_rows


def run_cleaning(project_root: Path | str, near_duplicate_threshold: int = 8) -> CleaningResult:
    """Create deterministic cleaning metadata using only ``data/extracted/v1/train``."""

    root = Path(project_root).resolve()
    scanned_records, integrity_rows = scan_v1_train(root)
    manifest_records, duplicate_rows, exact_conflicts = resolve_exact_duplicates(scanned_records)
    source_conflicts = detect_source_label_conflicts(manifest_records)
    retained_records = sorted(
        (record for record in manifest_records if record.status == "kept"),
        key=lambda item: item.source_relative_path,
    )
    near_pairs = find_near_duplicates(retained_records, near_duplicate_threshold)
    report_rows = build_cleaning_report(
        scanned_records, manifest_records, integrity_rows, duplicate_rows, source_conflicts, near_pairs
    )

    metadata_dir = root / "data" / "metadata"
    manifest_rows = []
    for record in sorted(manifest_records, key=lambda item: item.source_relative_path):
        row = asdict(record)
        row["source_path"] = record.source_relative_path
        manifest_rows.append(row)
    write_csv(metadata_dir / "cleaning_manifest.csv", manifest_rows)
    write_csv(metadata_dir / "cleaning_report.csv", report_rows)

    result = CleaningResult(
        project_root=root,
        source_root=get_v1_train_root(root),
        retained_records=tuple(retained_records),
        manifest_records=tuple(sorted(manifest_records, key=lambda item: item.source_relative_path)),
        report_rows=tuple(report_rows),
        critical_conflicts=tuple([*exact_conflicts, *source_conflicts]),
        near_duplicate_pairs=tuple(near_pairs),
    )
    LOGGER.info(
        "Cleaning scanned %d images; retained %d; exact exclusions %d; conflicts %d.",
        len(scanned_records), len(retained_records),
        sum(record.status == "excluded_exact_duplicate" for record in manifest_records),
        len(result.critical_conflicts),
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean only data/extracted/v1/train without modifying originals.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--near-duplicate-threshold", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_cleaning(arguments.project_root, arguments.near_duplicate_threshold)
    if not result.is_safe_to_split:
        LOGGER.error("Cleaning found critical label conflicts; splitting is blocked.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
