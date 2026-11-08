"""Deterministic, leakage-safe V1/train splitting with confirmed visual groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.data.data_audit import sha256_file, write_csv, write_json
from src.data.data_cleaning import (
    CleanedImageRecord,
    CleaningResult,
    CriticalDataConflictError,
    assert_v1_train_source,
    get_v1_train_root,
    run_cleaning,
)

LOGGER = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "val", "test_internal")
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test_internal": 0.10}

# Confirmed by TASK 02B review and TASK 02E user decisions. These are explicit
# decisions rather than relationships inferred from perceptual-hash distance alone.
CONFIRMED_CROSS_SPLIT_PAIR_IDS = frozenset({
    "ND-004", "ND-005", "ND-006", "ND-008", "ND-009", "ND-010", "ND-017",
    "ND-027", "ND-028", "ND-029", "ND-031", "ND-032", "ND-034", "ND-036",
    "ND-037", "ND-038", "ND-042", "ND-043", "ND-047", "ND-021",
    "ND-023", "ND-025",
})
CONFIRMED_SAME_SPLIT_PAIR_IDS = frozenset({
    "ND-001", "ND-002", "ND-007", "ND-011", "ND-013", "ND-016", "ND-018",
    "ND-024", "ND-026", "ND-030", "ND-033", "ND-035", "ND-039", "ND-040",
    "ND-041", "ND-045",
})
FALSE_POSITIVE_PAIR_IDS = frozenset({"ND-003", "ND-012", "ND-044", "ND-046"})


@dataclass(frozen=True)
class SplitAssignment:
    """A clean source record allocated to one deterministic destination split."""

    record: CleanedImageRecord
    split: str
    group_id: str
    processed_relative_path: str


@dataclass(frozen=True)
class ConfirmedVisualGroups:
    """Connected visual components built only from explicitly confirmed pair IDs."""

    pairs_by_id: dict[str, dict[str, str]]
    confirmed_pair_ids: frozenset[str]
    ambiguous_pair_ids: frozenset[str]
    false_positive_pair_ids: frozenset[str]


class DisjointSet:
    """Small deterministic union-find implementation for source and visual groups."""

    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    return sha256_file(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def _allocate_target_counts(total: int) -> dict[str, int]:
    """Allocate integer counts with the largest-remainder method."""

    raw = {split: total * ratio for split, ratio in SPLIT_RATIOS.items()}
    targets = {split: int(value) for split, value in raw.items()}
    remainder = total - sum(targets.values())
    ranked = sorted(
        SPLIT_NAMES,
        key=lambda split: (-(raw[split] - targets[split]), SPLIT_NAMES.index(split)),
    )
    for split in ranked[:remainder]:
        targets[split] += 1
    return targets


def _group_id_for_records(records: Sequence[CleanedImageRecord], visual_paths: set[str]) -> str:
    paths = sorted(record.source_relative_path for record in records)
    if any(path in visual_paths for path in paths):
        digest = hashlib.sha256("|".join(paths).encode("utf-8")).hexdigest()[:16]
        return f"visual:{digest}"
    source_identities = {record.source_identity for record in records if record.source_identity}
    if len(source_identities) == 1 and len(records) > 1:
        return f"source:{next(iter(source_identities))}"
    return f"sha256:{records[0].sha256}"


def load_confirmed_visual_groups(project_root: Path | str) -> ConfirmedVisualGroups:
    """Verify TASK 02B decisions against the V1-only review manifest."""

    root = Path(project_root).resolve()
    review_path = root / "data" / "metadata" / "near_duplicate_split_review.csv"
    rows = _read_csv(review_path)
    pairs_by_id = {row["pair_id"]: row for row in rows}
    expected_confirmed = CONFIRMED_CROSS_SPLIT_PAIR_IDS | CONFIRMED_SAME_SPLIT_PAIR_IDS
    missing = (expected_confirmed | FALSE_POSITIVE_PAIR_IDS) - set(pairs_by_id)
    if missing:
        raise ValueError(f"Near-duplicate review is missing expected pair IDs: {sorted(missing)}")
    if expected_confirmed & FALSE_POSITIVE_PAIR_IDS:
        raise ValueError("Confirmed and false-positive pair sets overlap.")

    for pair_id in expected_confirmed | FALSE_POSITIVE_PAIR_IDS:
        if pairs_by_id[pair_id]["manifest_mapping_status"] != "mapped":
            raise ValueError(f"Expected pair is no longer mapped by the V1 split manifest: {pair_id}")

    observed_ids = set(pairs_by_id)
    ambiguous = observed_ids - expected_confirmed - FALSE_POSITIVE_PAIR_IDS
    return ConfirmedVisualGroups(
        pairs_by_id=pairs_by_id,
        confirmed_pair_ids=frozenset(expected_confirmed),
        ambiguous_pair_ids=frozenset(ambiguous),
        false_positive_pair_ids=FALSE_POSITIVE_PAIR_IDS,
    )


def _build_record_groups(
    records: Sequence[CleanedImageRecord], confirmed_groups: ConfirmedVisualGroups | None
) -> tuple[list[tuple[str, list[CleanedImageRecord]]], dict[str, list[str]], set[str]]:
    """Join source identities and explicitly confirmed visual pairs into components."""

    by_path = {record.source_relative_path: record for record in records}
    if len(by_path) != len(records):
        raise ValueError("Cleaned records contain duplicate source paths.")
    disjoint_set = DisjointSet(by_path)

    by_source_identity: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.source_identity:
            by_source_identity[record.source_identity].append(record.source_relative_path)
    for paths in by_source_identity.values():
        for path in paths[1:]:
            disjoint_set.union(paths[0], path)

    visual_paths: set[str] = set()
    pair_ids_by_path: dict[str, list[str]] = defaultdict(list)
    if confirmed_groups is not None:
        for pair_id in sorted(confirmed_groups.confirmed_pair_ids):
            row = confirmed_groups.pairs_by_id[pair_id]
            left, right = row["left_source_relative_path"], row["right_source_relative_path"]
            if left not in by_path or right not in by_path:
                raise ValueError(f"Confirmed visual pair is absent from cleaned V1/train records: {pair_id}")
            disjoint_set.union(left, right)
            visual_paths.update((left, right))
            pair_ids_by_path[left].append(pair_id)
            pair_ids_by_path[right].append(pair_id)

    components: dict[str, list[CleanedImageRecord]] = defaultdict(list)
    for record in records:
        components[disjoint_set.find(record.source_relative_path)].append(record)
    groups: list[tuple[str, list[CleanedImageRecord]]] = []
    group_pair_ids: dict[str, list[str]] = {}
    for component_records in components.values():
        ordered_records = sorted(component_records, key=lambda item: item.source_relative_path)
        group_id = _group_id_for_records(ordered_records, visual_paths)
        groups.append((group_id, ordered_records))
        group_pair_ids[group_id] = sorted({
            pair_id for record in ordered_records for pair_id in pair_ids_by_path[record.source_relative_path]
        })
    return sorted(groups, key=lambda item: item[0]), group_pair_ids, visual_paths


def _group_class_counts(group_records: Sequence[CleanedImageRecord]) -> Counter[str]:
    return Counter(record.class_name for record in group_records)


def _assignment_cost(
    split: str,
    group_counts: Counter[str],
    actual: dict[str, Counter[str]],
    targets: dict[str, dict[str, int]],
) -> float:
    return sum(
        ((actual[class_name][split] + group_counts[class_name] - targets[class_name][split]) /
         max(1, targets[class_name][split])) ** 2
        for class_name in targets
    )


def _can_fit_without_exceeding(
    split: str,
    group_counts: Counter[str],
    actual: dict[str, Counter[str]],
    targets: dict[str, dict[str, int]],
) -> bool:
    return all(
        actual[class_name][split] + count <= targets[class_name][split]
        for class_name, count in group_counts.items()
    )


def stratified_group_split(
    records: Sequence[CleanedImageRecord],
    seed: int = 42,
    confirmed_groups: ConfirmedVisualGroups | None = None,
) -> list[SplitAssignment]:
    """Allocate source/visual components while minimizing per-class ratio deviation."""

    if not records:
        raise ValueError("Cannot split an empty cleaned dataset.")
    if any(record.status != "kept" for record in records):
        raise ValueError("Only retained cleaning records may be split.")

    groups, _, visual_paths = _build_record_groups(records, confirmed_groups)
    classes = sorted({record.class_name for record in records})
    targets = {
        class_name: _allocate_target_counts(sum(record.class_name == class_name for record in records))
        for class_name in classes
    }
    actual = {class_name: Counter() for class_name in classes}
    group_assignment: dict[str, str] = {}

    multi_record_groups = [group for group in groups if len(group[1]) > 1]
    singleton_groups = [group for group in groups if len(group[1]) == 1]
    ordered_multi = sorted(
        multi_record_groups,
        key=lambda item: (-len(item[1]), _stable_rank(item[0], seed)),
    )
    for group_id, group_records in ordered_multi:
        group_counts = _group_class_counts(group_records)
        fitting_splits = [
            split for split in SPLIT_NAMES
            if _can_fit_without_exceeding(split, group_counts, actual, targets)
        ]
        candidates = fitting_splits or list(SPLIT_NAMES)
        split = min(
            candidates,
            key=lambda candidate: (
                _assignment_cost(candidate, group_counts, actual, targets),
                SPLIT_NAMES.index(candidate),
            ),
        )
        group_assignment[group_id] = split
        for class_name, count in group_counts.items():
            actual[class_name][split] += count

    for group_id, group_records in sorted(singleton_groups, key=lambda item: _stable_rank(item[0], seed)):
        record = group_records[0]
        deficits = {
            split: targets[record.class_name][split] - actual[record.class_name][split]
            for split in SPLIT_NAMES
        }
        split = max(
            SPLIT_NAMES,
            key=lambda candidate: (deficits[candidate], -SPLIT_NAMES.index(candidate)),
        )
        group_assignment[group_id] = split
        actual[record.class_name][split] += 1

    assignments: list[SplitAssignment] = []
    for group_id, group_records in groups:
        split = group_assignment[group_id]
        for record in group_records:
            processed_relative_path = Path("data") / "processed" / split / record.class_name / record.filename
            assignments.append(SplitAssignment(record, split, group_id, processed_relative_path.as_posix()))
    return sorted(assignments, key=lambda item: item.record.source_relative_path)


def validate_assignments(assignments: Iterable[SplitAssignment], project_root: Path | str) -> dict[str, Any]:
    """Validate provenance, SHA, source identity, and visual group isolation."""

    root = Path(project_root).resolve()
    source_root = get_v1_train_root(root).resolve()
    items = list(assignments)
    source_paths = [item.record.source_relative_path for item in items]
    hashes_to_splits: dict[str, set[str]] = defaultdict(set)
    groups_to_splits: dict[str, set[str]] = defaultdict(set)
    invalid_sources: list[str] = []
    invalid_destinations: list[str] = []
    for item in items:
        try:
            assert_v1_train_source(item.record.source_path, root)
        except (ValueError, FileNotFoundError):
            invalid_sources.append(item.record.source_relative_path)
        if not item.record.source_path.resolve().is_relative_to(source_root):
            invalid_sources.append(item.record.source_relative_path)
        if not item.processed_relative_path.startswith(f"data/processed/{item.split}/"):
            invalid_destinations.append(item.processed_relative_path)
        hashes_to_splits[item.record.sha256].add(item.split)
        groups_to_splits[item.group_id].add(item.split)
    return {
        "record_count": len(items),
        "unique_source_paths": len(source_paths) == len(set(source_paths)),
        "duplicate_source_paths": sorted(path for path, count in Counter(source_paths).items() if count > 1),
        "no_sha256_across_splits": all(len(splits) == 1 for splits in hashes_to_splits.values()),
        "sha256_cross_split_groups": sorted(digest for digest, splits in hashes_to_splits.items() if len(splits) > 1),
        "source_groups_isolated": all(len(splits) == 1 for splits in groups_to_splits.values()),
        "source_groups_across_splits": sorted(group for group, splits in groups_to_splits.items() if len(splits) > 1),
        "allowed_source_only": not invalid_sources,
        "invalid_source_paths": sorted(set(invalid_sources)),
        "valid_destination_layout": not invalid_destinations,
        "invalid_destination_paths": sorted(set(invalid_destinations)),
    }


def _ensure_assignment_validation(validation: dict[str, Any]) -> None:
    required = (
        "unique_source_paths", "no_sha256_across_splits", "source_groups_isolated",
        "allowed_source_only", "valid_destination_layout",
    )
    if not all(validation[key] for key in required):
        raise CriticalDataConflictError(f"Pre-write split validation failed: {validation}")


def _clear_managed_split_directories(processed_root: Path) -> None:
    processed_root.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        split_directory = (processed_root / split).resolve()
        if not split_directory.is_relative_to(processed_root.resolve()):
            raise ValueError(f"Unsafe processed split directory: {split_directory}")
        if split_directory.exists():
            shutil.rmtree(split_directory)
        split_directory.mkdir(parents=True, exist_ok=True)


def materialize_splits(
    assignments: Sequence[SplitAssignment], project_root: Path | str, output_root: Path
) -> list[dict[str, Any]]:
    """Copy source images to a supplied staging/final root and verify each copy hash."""

    root = Path(project_root).resolve()
    output_root = output_root.resolve()
    destinations: set[Path] = set()
    manifest_rows: list[dict[str, Any]] = []
    for assignment in assignments:
        source = assignment.record.source_path
        assert_v1_train_source(source, root)
        destination = (output_root / assignment.split / assignment.record.class_name / assignment.record.filename).resolve()
        if not destination.is_relative_to(output_root) or destination in destinations:
            raise ValueError(f"Unsafe or colliding processed destination: {destination}")
        destinations.add(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_sha256 = sha256_file(destination)
        if copied_sha256 != assignment.record.sha256:
            raise IOError(f"Copy integrity failure: {source} -> {destination}")
        manifest_rows.append({
            "split": assignment.split,
            "class_name": assignment.record.class_name,
            "group_id": assignment.group_id,
            "source_relative_path": assignment.record.source_relative_path,
            "processed_relative_path": assignment.processed_relative_path,
            "filename": assignment.record.filename,
            "sha256": assignment.record.sha256,
            "perceptual_hash": assignment.record.perceptual_hash,
            "source_identity": assignment.record.source_identity,
            "copy_sha256_verified": True,
        })
    return sorted(manifest_rows, key=lambda row: row["source_relative_path"])


def validate_staged_files(manifest_rows: Sequence[dict[str, Any]], staging_root: Path) -> dict[str, Any]:
    """Confirm staged copies exist and match their manifest SHA-256 values."""

    missing: list[str] = []
    mismatched: list[str] = []
    for row in manifest_rows:
        path = staging_root / row["split"] / row["class_name"] / row["filename"]
        if not path.is_file():
            missing.append(path.as_posix())
        elif sha256_file(path) != row["sha256"]:
            mismatched.append(path.as_posix())
    return {
        "all_staged_files_present": not missing,
        "all_staged_hashes_match": not mismatched,
        "missing_staged_files": missing,
        "mismatched_staged_hashes": mismatched,
    }


def _build_split_summary(
    assignments: Sequence[SplitAssignment], validation: dict[str, Any], seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_split = Counter(assignment.split for assignment in assignments)
    by_class_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_class_total = Counter()
    for assignment in assignments:
        by_class_split[assignment.record.class_name][assignment.split] += 1
        by_class_total[assignment.record.class_name] += 1
    total = len(assignments)
    class_distribution: dict[str, Any] = {}
    for class_name, class_total in sorted(by_class_total.items()):
        targets = _allocate_target_counts(class_total)
        actual = {split: by_class_split[class_name][split] for split in SPLIT_NAMES}
        class_distribution[class_name] = {
            "source_count": class_total,
            "target_counts": targets,
            "actual_counts": actual,
            "deviation": {split: actual[split] - targets[split] for split in SPLIT_NAMES},
        }
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_dataset": "data/extracted/v1/train",
        "excluded_sources": ["data/extracted/v1/test", "data/extracted/v5"],
        "seed": seed,
        "requested_ratios": SPLIT_RATIOS,
        "total_images": total,
        "split_counts": {split: by_split[split] for split in SPLIT_NAMES},
        "actual_ratios": {split: by_split[split] / total if total else 0.0 for split in SPLIT_NAMES},
        "class_distribution": class_distribution,
        "validation": validation,
    }
    if extra:
        summary.update(extra)
    return summary


def _archive_existing_split_metadata(root: Path) -> dict[str, Any]:
    """Archive current manifest/summary by content hash before a controlled resplit."""

    metadata_dir = root / "data" / "metadata"
    manifest_path = metadata_dir / "split_manifest.csv"
    summary_path = metadata_dir / "split_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Current split manifest and summary must exist before leakage-safe resplitting.")
    manifest_checksum = _sha256_path(manifest_path)
    summary_checksum = _sha256_path(summary_path)
    review_path = metadata_dir / "near_duplicate_split_review.csv"
    manifest_archive = metadata_dir / f"split_manifest_pre_task_02c_{manifest_checksum[:12]}.csv"
    summary_archive = metadata_dir / f"split_summary_pre_task_02c_{summary_checksum[:12]}.json"
    if not manifest_archive.exists():
        shutil.copy2(manifest_path, manifest_archive)
    if not summary_archive.exists():
        shutil.copy2(summary_path, summary_archive)
    archived = {
        "manifest": {"path": manifest_archive.relative_to(root).as_posix(), "sha256": manifest_checksum},
        "summary": {"path": summary_archive.relative_to(root).as_posix(), "sha256": summary_checksum},
    }
    if review_path.is_file():
        review_checksum = _sha256_path(review_path)
        review_archive = metadata_dir / f"near_duplicate_split_review_pre_task_02c_{review_checksum[:12]}.csv"
        if not review_archive.exists():
            shutil.copy2(review_path, review_archive)
        archived["near_duplicate_review"] = {
            "path": review_archive.relative_to(root).as_posix(), "sha256": review_checksum
        }
    return archived


def _commit_staging(root: Path, staging_root: Path) -> None:
    """Swap validated staged splits while retaining an in-process rollback path."""

    data_root = root / "data"
    processed_root = data_root / "processed"
    backup_root = data_root / ".processed_pre_task_02c_backup"
    allowed_existing = set(SPLIT_NAMES)
    if processed_root.exists():
        unexpected = {path.name for path in processed_root.iterdir()} - allowed_existing
        if unexpected:
            raise RuntimeError(f"Refusing to replace processed data with unexpected entries: {sorted(unexpected)}")
    if backup_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing rollback directory: {backup_root}")
    try:
        if processed_root.exists():
            shutil.move(str(processed_root), str(backup_root))
        shutil.move(str(staging_root), str(processed_root))
    except Exception:
        if not processed_root.exists() and backup_root.exists():
            shutil.move(str(backup_root), str(processed_root))
        raise
    if backup_root.exists():
        shutil.rmtree(backup_root)


def _near_duplicate_group_rows(
    assignments: Sequence[SplitAssignment], confirmed_groups: ConfirmedVisualGroups,
) -> list[dict[str, Any]]:
    by_group: dict[str, list[SplitAssignment]] = defaultdict(list)
    for assignment in assignments:
        if assignment.group_id.startswith("visual:"):
            by_group[assignment.group_id].append(assignment)
    rows: list[dict[str, Any]] = []
    for group_id, members in sorted(by_group.items()):
        member_paths = {member.record.source_relative_path for member in members}
        pair_ids = sorted(
            pair_id for pair_id in confirmed_groups.confirmed_pair_ids
            if {
                confirmed_groups.pairs_by_id[pair_id]["left_source_relative_path"],
                confirmed_groups.pairs_by_id[pair_id]["right_source_relative_path"],
            }.issubset(member_paths)
        )
        for member in sorted(members, key=lambda item: item.record.source_relative_path):
            rows.append({
                "visual_group_id": group_id,
                "member_count": len(members),
                "confirmed_pair_ids": pair_ids,
                "source_relative_path": member.record.source_relative_path,
                "class_name": member.record.class_name,
                "assigned_split": member.split,
                "source_identity": member.record.source_identity,
                "sha256": member.record.sha256,
                "group_status": "confirmed_visual_relationship",
            })
    return rows


def run_splitting(project_root: Path | str, seed: int = 42) -> dict[str, Any]:
    """Run the original source-identity-aware V1/train split without visual review groups."""

    root = Path(project_root).resolve()
    cleaning_result: CleaningResult = run_cleaning(root)
    if not cleaning_result.is_safe_to_split:
        raise CriticalDataConflictError("Cleaning found critical label conflicts. Split outputs were not written.")
    assignments = stratified_group_split(cleaning_result.retained_records, seed=seed)
    validation = validate_assignments(assignments, root)
    _ensure_assignment_validation(validation)
    processed_root = root / "data" / "processed"
    _clear_managed_split_directories(processed_root)
    manifest_rows = materialize_splits(assignments, root, processed_root)
    staged_validation = validate_staged_files(manifest_rows, processed_root)
    if not all(staged_validation[key] for key in ("all_staged_files_present", "all_staged_hashes_match")):
        raise CriticalDataConflictError(f"Post-copy split validation failed: {staged_validation}")
    summary = _build_split_summary(assignments, validation, seed, {"copy_validation": staged_validation})
    metadata_dir = root / "data" / "metadata"
    write_csv(metadata_dir / "split_manifest.csv", manifest_rows)
    write_json(metadata_dir / "split_summary.json", summary)
    LOGGER.info("Split %d images: train=%d, val=%d, test_internal=%d.",
                len(assignments), summary["split_counts"]["train"],
                summary["split_counts"]["val"], summary["split_counts"]["test_internal"])
    return summary


def run_leakage_safe_resplitting(project_root: Path | str, seed: int = 42) -> dict[str, Any]:
    """Resplit V1/train using only confirmed visual groups after staging validation."""

    root = Path(project_root).resolve()
    cleaning_result = run_cleaning(root)
    if not cleaning_result.is_safe_to_split:
        raise CriticalDataConflictError("Cleaning found critical label conflicts. Existing split was preserved.")
    confirmed_groups = load_confirmed_visual_groups(root)
    archived = _archive_existing_split_metadata(root)
    assignments = stratified_group_split(cleaning_result.retained_records, seed, confirmed_groups)
    validation = validate_assignments(assignments, root)
    _ensure_assignment_validation(validation)

    staging_root = root / "data" / ".processed_task_02c_staging"
    if staging_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing staging directory: {staging_root}")
    staging_root.mkdir(parents=True)
    try:
        manifest_rows = materialize_splits(assignments, root, staging_root)
        staging_validation = validate_staged_files(manifest_rows, staging_root)
        if not all(staging_validation[key] for key in ("all_staged_files_present", "all_staged_hashes_match")):
            raise CriticalDataConflictError(f"Staging validation failed; existing split preserved: {staging_validation}")
        _commit_staging(root, staging_root)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    group_rows = _near_duplicate_group_rows(assignments, confirmed_groups)
    visual_group_count = len({row["visual_group_id"] for row in group_rows})
    summary = _build_split_summary(assignments, validation, seed, {
        "copy_validation": staging_validation,
        "resplitting_mode": "confirmed_visual_group_aware",
        "confirmed_visual_pair_count": len(confirmed_groups.confirmed_pair_ids),
        "confirmed_visual_group_count": visual_group_count,
        "ambiguous_near_duplicate_pair_ids": sorted(confirmed_groups.ambiguous_pair_ids),
        "false_positive_near_duplicate_pair_ids": sorted(confirmed_groups.false_positive_pair_ids),
        "archived_previous_split": archived,
    })
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_dataset": "data/extracted/v1/train",
        "seed": seed,
        "requested_ratios": SPLIT_RATIOS,
        "archived_previous_split": archived,
        "confirmed_visual_pair_ids": sorted(confirmed_groups.confirmed_pair_ids),
        "confirmed_visual_pair_count": len(confirmed_groups.confirmed_pair_ids),
        "confirmed_visual_group_count": visual_group_count,
        "ambiguous_pair_ids": sorted(confirmed_groups.ambiguous_pair_ids),
        "ambiguous_pair_count": len(confirmed_groups.ambiguous_pair_ids),
        "false_positive_pair_ids": sorted(confirmed_groups.false_positive_pair_ids),
        "validation": validation,
        "staging_validation": staging_validation,
        "new_split_counts": summary["split_counts"],
        "new_actual_ratios": summary["actual_ratios"],
        "status": "completed",
        "notes": [
            "Only explicitly confirmed visual relationships were grouped.",
            "Ambiguous perceptual-hash candidates were not grouped automatically.",
            "Original V1/train images were copied, never modified.",
        ],
    }
    metadata_dir = root / "data" / "metadata"
    write_csv(metadata_dir / "split_manifest.csv", manifest_rows)
    write_json(metadata_dir / "split_summary.json", summary)
    write_csv(metadata_dir / "near_duplicate_groups.csv", group_rows)
    write_json(metadata_dir / "resplitting_report.json", report)
    LOGGER.info("Leakage-safe resplit complete: %d confirmed visual groups, train=%d, val=%d, test_internal=%d.",
                visual_group_count, summary["split_counts"]["train"],
                summary["split_counts"]["val"], summary["split_counts"]["test_internal"])
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split V1/train images with deterministic group isolation.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leakage-safe", action="store_true", help="Use confirmed TASK 02B visual groups and staging.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if arguments.leakage_safe:
        run_leakage_safe_resplitting(arguments.project_root, arguments.seed)
    else:
        run_splitting(arguments.project_root, arguments.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
