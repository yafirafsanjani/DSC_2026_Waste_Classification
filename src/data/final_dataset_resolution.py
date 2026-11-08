"""Resolve TASK 02D decisions without modifying V1/train source images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from src.data.data_audit import sha256_file, write_csv, write_json
from src.data.data_cleaning import CleanedImageRecord, CriticalDataConflictError, run_cleaning
from src.data.data_split import (
    ConfirmedVisualGroups,
    SPLIT_NAMES,
    SPLIT_RATIOS,
    _build_split_summary,
    _ensure_assignment_validation,
    _near_duplicate_group_rows,
    load_confirmed_visual_groups,
    materialize_splits,
    stratified_group_split,
    validate_assignments,
    validate_staged_files,
)

LOGGER = logging.getLogger(__name__)

MANUAL_DECISIONS = {
    "ND-021": "same_source_or_derivative",
    "ND-023": "same_source_or_derivative",
    "ND-025": "same_source_or_derivative",
    "ND-002": "label_noise_confirmed",
}
DECISION_RATIONALES = {
    "ND-021": (
        "User review: the beverage-can composition is highly similar despite color and watermark differences; "
        "treat the pair as one visual group to prevent potential leakage."
    ),
    "ND-023": (
        "User review: the beverage-can composition is highly similar despite color and watermark differences; "
        "treat the pair as one visual group to prevent potential leakage."
    ),
    "ND-025": (
        "User review: the beverage-can composition is highly similar despite color and watermark differences; "
        "treat the pair as one visual group to prevent potential leakage."
    ),
    "ND-002": (
        "User review: visually identical object is assigned the conflicting labels cardboard and paper; "
        "exclude both records from the active dataset pending label verification."
    ),
}
LABEL_NOISE_PAIR_ID = "ND-002"
LABEL_NOISE_SOURCE_PATHS = frozenset({
    "data/extracted/v1/train/cardboard/1368_jpg.rf.1ca9d9e39e2ab975d0371ac6387ada31.jpg",
    "data/extracted/v1/train/paper/1548_jpg.rf.097efc120796cf28c589550dab63ec5a.jpg",
})


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def record_task_02e_decisions(project_root: Path | str) -> dict[str, dict[str, str]]:
    """Persist the user-provided TASK 02E decisions in the review template."""

    root = Path(project_root).resolve()
    review_path = root / "data" / "metadata" / "final_manual_review.csv"
    rows, fieldnames = _read_csv(review_path)
    required_columns = {"pair_id", "user_decision", "user_rationale", "reviewer_name", "reviewed_at"}
    if not required_columns.issubset(fieldnames):
        raise ValueError(f"Final manual review is missing required columns: {sorted(required_columns - set(fieldnames))}")

    rows_by_id = {row["pair_id"]: row for row in rows}
    missing = set(MANUAL_DECISIONS) - set(rows_by_id)
    if missing:
        raise ValueError(f"Final manual review is missing decision rows: {sorted(missing)}")

    reviewed_at = datetime.now(UTC).isoformat()
    for pair_id, decision in MANUAL_DECISIONS.items():
        row = rows_by_id[pair_id]
        permitted = set(filter(None, row.get("user_decision_options", "").split("|")))
        if decision not in permitted:
            raise ValueError(f"Decision {decision} is not permitted for {pair_id}.")
        existing = row.get("user_decision", "")
        if existing and existing != decision:
            raise CriticalDataConflictError(
                f"Existing manual decision for {pair_id} conflicts with TASK 02E: {existing}"
            )
        row["user_decision"] = decision
        row["user_rationale"] = DECISION_RATIONALES[pair_id]
        row["reviewer_name"] = "user_visual_review"
        row["reviewed_at"] = row.get("reviewed_at") or reviewed_at
    _write_csv(review_path, rows, fieldnames)
    return {pair_id: rows_by_id[pair_id] for pair_id in MANUAL_DECISIONS}


def _exclude_label_noise_records(
    records: Sequence[CleanedImageRecord],
    manual_rows: dict[str, dict[str, str]],
    excluded_source_paths: frozenset[str] = LABEL_NOISE_SOURCE_PATHS,
) -> tuple[list[CleanedImageRecord], list[dict[str, Any]]]:
    """Keep raw label-noise images traceable while removing them from active splits."""

    records_by_path = {record.source_relative_path: record for record in records}
    missing = excluded_source_paths - set(records_by_path)
    if missing:
        raise CriticalDataConflictError(
            f"Confirmed ND-002 label-noise sources are absent from cleaned V1/train records: {sorted(missing)}"
        )
    if manual_rows[LABEL_NOISE_PAIR_ID]["user_decision"] != "label_noise_confirmed":
        raise CriticalDataConflictError("ND-002 must be marked label_noise_confirmed before exclusion.")

    excluded: list[dict[str, Any]] = []
    active_records: list[CleanedImageRecord] = []
    for record in records:
        if record.source_relative_path not in excluded_source_paths:
            active_records.append(record)
            continue
        side = (
            "left" if record.source_relative_path == manual_rows[LABEL_NOISE_PAIR_ID]["left_source_relative_path"] else "right"
        )
        excluded.append({
            "pair_id": LABEL_NOISE_PAIR_ID,
            "source_relative_path": record.source_relative_path,
            "filename": record.filename,
            "class_name": record.class_name,
            "source_identity": record.source_identity,
            "sha256": record.sha256,
            "perceptual_hash": record.perceptual_hash,
            "prior_split": manual_rows[LABEL_NOISE_PAIR_ID][f"{side}_split"],
            "reason": "confirmed_cross_class_visual_label_noise",
            "decision": "label_noise_confirmed",
            "active_dataset_status": "excluded_pending_label_verification",
        })
    return active_records, sorted(excluded, key=lambda row: row["source_relative_path"])


def _active_confirmed_groups(confirmed_groups: ConfirmedVisualGroups) -> ConfirmedVisualGroups:
    """Remove only the excluded ND-002 relation from active splitting components."""

    if LABEL_NOISE_PAIR_ID not in confirmed_groups.confirmed_pair_ids:
        raise CriticalDataConflictError("ND-002 is not registered as a confirmed visual pair.")
    return ConfirmedVisualGroups(
        pairs_by_id=confirmed_groups.pairs_by_id,
        confirmed_pair_ids=frozenset(confirmed_groups.confirmed_pair_ids - {LABEL_NOISE_PAIR_ID}),
        ambiguous_pair_ids=confirmed_groups.ambiguous_pair_ids,
        false_positive_pair_ids=confirmed_groups.false_positive_pair_ids,
    )


def _archive_current_metadata(project_root: Path) -> dict[str, dict[str, str]]:
    """Archive all active TASK 02 metadata before replacing split outputs."""

    metadata_dir = project_root / "data" / "metadata"
    names = (
        "split_manifest.csv",
        "split_summary.json",
        "near_duplicate_groups.csv",
        "near_duplicate_split_review.csv",
        "resplitting_report.json",
        "final_manual_review.csv",
    )
    archived: dict[str, dict[str, str]] = {}
    for name in names:
        source_path = metadata_dir / name
        if not source_path.is_file():
            continue
        checksum = sha256_file(source_path)
        archive_path = metadata_dir / f"{source_path.stem}_pre_task_02e_{checksum[:12]}{source_path.suffix}"
        if not archive_path.exists():
            shutil.copy2(source_path, archive_path)
        archived[source_path.stem] = {
            "path": archive_path.relative_to(project_root).as_posix(),
            "sha256": checksum,
        }
    return archived


def _commit_staging_with_rollback(
    project_root: Path, staging_root: Path, manifest_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Promote only a verified staging tree and restore the old tree on failure."""

    data_root = project_root / "data"
    processed_root = data_root / "processed"
    backup_root = data_root / ".processed_pre_task_02e_backup"
    allowed_existing = set(SPLIT_NAMES)
    if processed_root.exists():
        unexpected = {path.name for path in processed_root.iterdir()} - allowed_existing
        if unexpected:
            raise RuntimeError(f"Refusing to replace processed data with unexpected entries: {sorted(unexpected)}")
    if backup_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing rollback directory: {backup_root}")

    promoted = False
    try:
        if processed_root.exists():
            shutil.move(str(processed_root), str(backup_root))
        shutil.move(str(staging_root), str(processed_root))
        promoted = True
        committed_validation = validate_staged_files(manifest_rows, processed_root)
        if not all(committed_validation[key] for key in ("all_staged_files_present", "all_staged_hashes_match")):
            raise CriticalDataConflictError(f"Committed split validation failed: {committed_validation}")
    except Exception:
        if promoted and processed_root.exists():
            shutil.rmtree(processed_root)
        if backup_root.exists() and not processed_root.exists():
            shutil.move(str(backup_root), str(processed_root))
        raise
    if backup_root.exists():
        shutil.rmtree(backup_root)
    return committed_validation


def _source_hash_validation(records: Sequence[CleanedImageRecord]) -> dict[str, Any]:
    mismatches = [
        record.source_relative_path
        for record in records
        if sha256_file(record.source_path) != record.sha256
    ]
    return {
        "source_files_unchanged": not mismatches,
        "source_sha256_mismatches": sorted(mismatches),
    }


def _excluded_group_rows(
    excluded_records: Sequence[dict[str, Any]], confirmed_groups: ConfirmedVisualGroups
) -> list[dict[str, Any]]:
    """Retain ND-002 membership in group metadata with an explicit inactive status."""

    pair = confirmed_groups.pairs_by_id[LABEL_NOISE_PAIR_ID]
    pair_paths = sorted((pair["left_source_relative_path"], pair["right_source_relative_path"]))
    group_digest = hashlib.sha256("|".join(pair_paths).encode("utf-8")).hexdigest()[:16]
    group_id = f"visual:{group_digest}"
    return [
        {
            "visual_group_id": group_id,
            "member_count": len(excluded_records),
            "confirmed_pair_ids": [LABEL_NOISE_PAIR_ID],
            "source_relative_path": record["source_relative_path"],
            "class_name": record["class_name"],
            "assigned_split": "excluded",
            "source_identity": record["source_identity"],
            "sha256": record["sha256"],
            "group_status": "excluded_confirmed_label_noise",
        }
        for record in excluded_records
    ]


def run_final_dataset_resolution(project_root: Path | str, seed: int = 42) -> dict[str, Any]:
    """Apply user decisions, exclude ND-002, and write a leakage-safe active split."""

    root = Path(project_root).resolve()
    manual_rows = record_task_02e_decisions(root)
    cleaning_result = run_cleaning(root)
    if not cleaning_result.is_safe_to_split:
        raise CriticalDataConflictError("Cleaning found critical label conflicts. Existing split was preserved.")

    active_records, excluded_records = _exclude_label_noise_records(
        cleaning_result.retained_records, manual_rows
    )
    all_confirmed_groups = load_confirmed_visual_groups(root)
    active_confirmed_groups = _active_confirmed_groups(all_confirmed_groups)
    assignments = stratified_group_split(active_records, seed, active_confirmed_groups)
    validation = validate_assignments(assignments, root)
    validation.update(_source_hash_validation(cleaning_result.retained_records))
    _ensure_assignment_validation(validation)
    if not validation["source_files_unchanged"]:
        raise CriticalDataConflictError(f"Source images changed unexpectedly: {validation}")

    archived = _archive_current_metadata(root)
    staging_root = root / "data" / ".processed_task_02e_staging"
    if staging_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing staging directory: {staging_root}")
    staging_root.mkdir(parents=True)
    try:
        manifest_rows = materialize_splits(assignments, root, staging_root)
        staging_validation = validate_staged_files(manifest_rows, staging_root)
        if not all(staging_validation[key] for key in ("all_staged_files_present", "all_staged_hashes_match")):
            raise CriticalDataConflictError(f"Staging validation failed; existing split preserved: {staging_validation}")
        committed_validation = _commit_staging_with_rollback(root, staging_root, manifest_rows)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    active_group_rows = _near_duplicate_group_rows(assignments, active_confirmed_groups)
    group_rows = active_group_rows + _excluded_group_rows(excluded_records, all_confirmed_groups)
    active_visual_group_count = len({row["visual_group_id"] for row in active_group_rows})
    summary = _build_split_summary(assignments, validation, seed, {
        "copy_validation": committed_validation,
        "staging_validation": staging_validation,
        "resplitting_mode": "task_02e_manual_resolution_group_aware",
        "source_images_before_label_noise_exclusion": len(cleaning_result.retained_records),
        "excluded_label_noise_count": len(excluded_records),
        "excluded_label_noise_records": excluded_records,
        "confirmed_visual_pair_count": len(all_confirmed_groups.confirmed_pair_ids),
        "active_confirmed_visual_pair_count": len(active_confirmed_groups.confirmed_pair_ids),
        "active_confirmed_visual_group_count": active_visual_group_count,
        "archived_previous_split": archived,
        "manual_review_decisions": MANUAL_DECISIONS,
    })
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed",
        "source_dataset": "data/extracted/v1/train",
        "seed": seed,
        "requested_ratios": SPLIT_RATIOS,
        "archived_previous_split": archived,
        "manual_review_decisions": MANUAL_DECISIONS,
        "confirmed_visual_pair_ids": sorted(all_confirmed_groups.confirmed_pair_ids),
        "confirmed_visual_pair_count": len(all_confirmed_groups.confirmed_pair_ids),
        "active_confirmed_visual_pair_ids": sorted(active_confirmed_groups.confirmed_pair_ids),
        "active_confirmed_visual_pair_count": len(active_confirmed_groups.confirmed_pair_ids),
        "active_confirmed_visual_group_count": active_visual_group_count,
        "ambiguous_pair_ids": sorted(active_confirmed_groups.ambiguous_pair_ids),
        "false_positive_pair_ids": sorted(active_confirmed_groups.false_positive_pair_ids),
        "source_images_before_label_noise_exclusion": len(cleaning_result.retained_records),
        "active_image_count": len(assignments),
        "excluded_label_noise": {
            "pair_id": LABEL_NOISE_PAIR_ID,
            "count": len(excluded_records),
            "reason": "confirmed_cross_class_visual_label_noise",
            "records": excluded_records,
        },
        "validation": validation,
        "staging_validation": staging_validation,
        "committed_validation": committed_validation,
        "new_split_counts": summary["split_counts"],
        "new_actual_ratios": summary["actual_ratios"],
        "notes": [
            "ND-021, ND-023, and ND-025 were promoted to confirmed visual relationships by user review.",
            "ND-002 raw images and original labels were preserved but excluded from all active splits.",
            "Only V1/train source images were read and copied; V1/test and V5 were excluded.",
        ],
    }
    metadata_dir = root / "data" / "metadata"
    write_csv(metadata_dir / "split_manifest.csv", manifest_rows)
    write_json(metadata_dir / "split_summary.json", summary)
    write_csv(metadata_dir / "near_duplicate_groups.csv", group_rows)
    write_json(metadata_dir / "resplitting_report.json", report)
    LOGGER.info(
        "TASK 02E active split complete: %d active images, %d excluded label-noise records.",
        len(assignments),
        len(excluded_records),
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve TASK 02E manual dataset-review decisions.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_final_dataset_resolution(arguments.project_root, arguments.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
