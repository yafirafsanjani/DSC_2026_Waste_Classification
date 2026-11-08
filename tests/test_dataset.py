from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from src.data.data_audit import (
    extract_roboflow_source_identity,
    hamming_distance,
    run_audit,
)


def _write_image(path: Path, color: tuple[int, int, int], quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 48), color=color)
    image.save(path, format="JPEG", quality=quality)


def test_extract_roboflow_source_identity() -> None:
    source, roboflow_id = extract_roboflow_source_identity(
        "023_jpg.rf.b65c870a4cc0307ba59b45188c5a7675.jpg"
    )

    assert source == "023_jpg"
    assert roboflow_id == "b65c870a4cc0307ba59b45188c5a7675"
    assert extract_roboflow_source_identity("plain_image.jpg") == (None, None)
    assert hamming_distance("0000000000000000", "000000000000000f") == 4


def test_run_audit_reports_integrity_duplicates_and_leakage(tmp_path: Path) -> None:
    v1_train = tmp_path / "data" / "extracted" / "v1" / "train" / "cardboard"
    v1_test = tmp_path / "data" / "extracted" / "v1" / "test" / "cardboard"
    v5_train = tmp_path / "data" / "extracted" / "v5" / "train" / "cardboard"
    v5_valid = tmp_path / "data" / "extracted" / "v5" / "valid" / "paper"

    original = v1_train / "shared_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg"
    _write_image(original, (220, 30, 30), quality=95)
    original_bytes = original.read_bytes()

    exact_copy = v5_train / "copied_jpg.rf.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.jpg"
    exact_copy.parent.mkdir(parents=True, exist_ok=True)
    exact_copy.write_bytes(original_bytes)

    _write_image(
        v1_test / "restricted_jpg.rf.cccccccccccccccccccccccccccccccc.jpg",
        (30, 220, 30),
        quality=95,
    )
    _write_image(
        v5_valid / "restricted_jpg.rf.dddddddddddddddddddddddddddddddd.jpg",
        (35, 220, 35),
        quality=70,
    )
    _write_image(v1_train / "near_left.jpg", (30, 30, 220), quality=95)
    _write_image(v5_train / "near_right.jpg", (30, 30, 220), quality=55)

    (v1_train / "not_an_image.txt").write_text("not an image", encoding="utf-8")
    (v1_train / "corrupt.jpg").write_text("not a jpeg", encoding="utf-8")

    summary = run_audit(tmp_path, near_duplicate_threshold=8)

    assert original.read_bytes() == original_bytes
    assert summary["totals"]["readable_images"] == 6
    assert summary["integrity_issue_counts"]["invalid_image_extension"] == 1
    assert summary["integrity_issue_counts"]["unreadable_image"] == 1
    assert summary["totals"]["exact_duplicate_groups"] == 1
    assert summary["totals"]["near_duplicate_pairs"] >= 1
    assert summary["totals"]["cross_context_leakage_findings"] >= 2
    assert summary["class_comparison"]["only_v1"] == []
    assert summary["class_comparison"]["only_v5"] == ["paper"]

    metadata_dir = tmp_path / "data" / "metadata"
    expected_reports = {
        "dataset_audit_image_manifest.csv",
        "dataset_audit_distribution.csv",
        "dataset_audit_integrity_issues.csv",
        "dataset_audit_exact_duplicates.csv",
        "dataset_audit_near_duplicates.csv",
        "dataset_audit_source_identity_groups.csv",
        "dataset_audit_leakage.csv",
        "dataset_audit_summary.json",
    }
    assert {path.name for path in metadata_dir.iterdir()} == expected_reports
    assert (tmp_path / "outputs" / "figures" / "dataset_class_distribution.png").is_file()

    with (metadata_dir / "dataset_audit_leakage.csv").open(encoding="utf-8", newline="") as file_handle:
        leakage_rows = list(csv.DictReader(file_handle))
    assert any(row["evidence_type"] == "roboflow_source_identity" for row in leakage_rows)

    with (metadata_dir / "dataset_audit_summary.json").open(encoding="utf-8") as file_handle:
        persisted_summary = json.load(file_handle)
    assert persisted_summary["totals"] == summary["totals"]

import pytest

from src.data.data_cleaning import CriticalDataConflictError, run_cleaning
from src.data.data_split import run_splitting


def _roboflow_name(source: str, number: int) -> str:
    return f"{source}.rf.{number:032x}.jpg"


def _write_training_image(root: Path, class_name: str, source: str, number: int, color: tuple[int, int, int]) -> Path:
    path = root / "data" / "extracted" / "v1" / "train" / class_name / _roboflow_name(source, number)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 48), color=color)
    image.putpixel((number % 48, (number // 48) % 48), (255, 255, 255))
    image.save(path, format="JPEG", quality=95)
    return path


def test_cleaning_and_group_aware_split_are_reproducible(tmp_path: Path) -> None:
    for index in range(10):
        _write_training_image(tmp_path, "cardboard", f"cardboard_{index}_jpg", index + 1, (20 + index, 60, 180))
        _write_training_image(tmp_path, "paper", f"paper_{index}_jpg", index + 101, (180, 60 + index, 20))

    _write_training_image(tmp_path, "cardboard", "shared_source_jpg", 201, (60, 140, 40))
    _write_training_image(tmp_path, "cardboard", "shared_source_jpg", 202, (60, 145, 40))
    exact_left = _write_training_image(tmp_path, "cardboard", "exact_left_jpg", 301, (200, 40, 40))
    exact_right = tmp_path / "data" / "extracted" / "v1" / "train" / "cardboard" / _roboflow_name("exact_right_jpg", 302)
    exact_right.write_bytes(exact_left.read_bytes())

    # Sentinels must never be included because TASK 02 admits only V1/train.
    _write_image(tmp_path / "data" / "extracted" / "v1" / "test" / "paper" / "forbidden.jpg", (1, 2, 3))
    _write_image(tmp_path / "data" / "extracted" / "v5" / "train" / "paper" / "forbidden.jpg", (3, 2, 1))

    cleaning = run_cleaning(tmp_path)
    assert cleaning.is_safe_to_split
    assert len(cleaning.manifest_records) == 24
    assert len(cleaning.retained_records) == 23
    assert sum(record.status == "excluded_exact_duplicate" for record in cleaning.manifest_records) == 1
    assert all("/test/" not in record.source_relative_path for record in cleaning.retained_records)
    assert all("/v5/" not in record.source_relative_path for record in cleaning.retained_records)

    first_summary = run_splitting(tmp_path, seed=42)
    first_manifest = (tmp_path / "data" / "metadata" / "split_manifest.csv").read_bytes()
    second_summary = run_splitting(tmp_path, seed=42)
    second_manifest = (tmp_path / "data" / "metadata" / "split_manifest.csv").read_bytes()

    assert first_summary["total_images"] == 23
    assert first_summary["split_counts"] == second_summary["split_counts"]
    assert first_manifest == second_manifest
    validation = first_summary["validation"]
    assert validation["unique_source_paths"]
    assert validation["no_sha256_across_splits"]
    assert validation["source_groups_isolated"]
    assert validation["allowed_source_only"]
    assert validation["valid_destination_layout"]
    assert not validation["duplicate_source_paths"]
    assert not validation["sha256_cross_split_groups"]
    assert not validation["source_groups_across_splits"]
    assert not validation["invalid_source_paths"]
    assert not validation["invalid_destination_paths"]

    with (tmp_path / "data" / "metadata" / "split_manifest.csv").open(encoding="utf-8", newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))
    assert len(rows) == 23
    assert {row["split"] for row in rows} == {"train", "val", "test_internal"}
    for class_name in {row["class_name"] for row in rows}:
        assert {row["split"] for row in rows if row["class_name"] == class_name} == {
            "train", "val", "test_internal"
        }
    assert sum(1 for _ in (tmp_path / "data" / "processed").rglob("*.jpg")) == len(rows)
    assert all(row["source_relative_path"].startswith("data/extracted/v1/train/") for row in rows)
    assert all("/v1/test/" not in row["source_relative_path"] for row in rows)
    assert all("/v5/" not in row["source_relative_path"] for row in rows)

    shared_rows = [row for row in rows if row["group_id"] == "source:shared_source_jpg"]
    assert len(shared_rows) == 2
    assert len({row["split"] for row in shared_rows}) == 1
    assert all((tmp_path / row["processed_relative_path"]).is_file() for row in rows)


def test_split_blocks_cross_class_exact_duplicate_conflict(tmp_path: Path) -> None:
    first = _write_training_image(tmp_path, "cardboard", "conflict_a_jpg", 1, (90, 90, 90))
    second = tmp_path / "data" / "extracted" / "v1" / "train" / "paper" / _roboflow_name("conflict_b_jpg", 2)
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_bytes(first.read_bytes())

    cleaning = run_cleaning(tmp_path)
    assert not cleaning.is_safe_to_split
    assert cleaning.critical_conflicts[0]["issue_type"] == "exact_duplicate_label_conflict"
    with pytest.raises(CriticalDataConflictError):
        run_splitting(tmp_path, seed=42)
    assert not (tmp_path / "data" / "processed").exists()

from src.data.near_duplicate_review import ManifestRecord, build_review_rows, split_combination


def test_near_duplicate_review_flags_cross_split_candidates() -> None:
    left_path = "data/extracted/v1/train/cardboard/left.jpg"
    right_path = "data/extracted/v1/train/paper/right.jpg"
    manifest = {
        left_path: ManifestRecord(
            source_relative_path=left_path,
            processed_relative_path="data/processed/train/cardboard/left.jpg",
            split="train",
            class_name="cardboard",
            source_identity="left_source",
            sha256="a" * 64,
            perceptual_hash="0000000000000000",
        ),
        right_path: ManifestRecord(
            source_relative_path=right_path,
            processed_relative_path="data/processed/val/paper/right.jpg",
            split="val",
            class_name="paper",
            source_identity="right_source",
            sha256="b" * 64,
            perceptual_hash="0000000000000000",
        ),
    }
    rows = build_review_rows(
        [{"left_path": left_path, "right_path": right_path, "hamming_distance": "0"}],
        manifest,
    )

    assert split_combination("val", "train") == "TrainValidation"
    assert rows[0]["split_combination"] == "TrainValidation"
    assert rows[0]["perceptual_hash_distance_verified"]
    assert rows[0]["cross_split"]
    assert rows[0]["manual_review_required"]
    assert rows[0]["leakage_risk"] == "review"
    assert rows[0]["relationship_assessment"] == "very_high_visual_similarity_distinct_source_metadata"

from src.data.data_split import (
    ConfirmedVisualGroups,
    _ensure_assignment_validation,
    stratified_group_split,
)


def test_confirmed_visual_group_stays_in_one_split_and_validation_fails_safe(tmp_path: Path) -> None:
    _write_training_image(tmp_path, "cardboard", "visual_left_jpg", 1, (30, 40, 50))
    _write_training_image(tmp_path, "paper", "visual_right_jpg", 2, (60, 70, 80))
    for index in range(12):
        _write_training_image(tmp_path, "cardboard", f"card_{index}_jpg", index + 10, (index + 10, 100, 30))
        _write_training_image(tmp_path, "paper", f"paper_{index}_jpg", index + 30, (30, index + 30, 100))

    cleaning = run_cleaning(tmp_path)
    left, right = cleaning.retained_records[0], cleaning.retained_records[-1]
    groups = ConfirmedVisualGroups(
        pairs_by_id={
            "ND-001": {
                "left_source_relative_path": left.source_relative_path,
                "right_source_relative_path": right.source_relative_path,
            }
        },
        confirmed_pair_ids=frozenset({"ND-001"}),
        ambiguous_pair_ids=frozenset(),
        false_positive_pair_ids=frozenset(),
    )
    assignments = stratified_group_split(cleaning.retained_records, seed=42, confirmed_groups=groups)
    repeat_assignments = stratified_group_split(cleaning.retained_records, seed=42, confirmed_groups=groups)
    assert [(item.record.source_relative_path, item.split) for item in assignments] == [
        (item.record.source_relative_path, item.split) for item in repeat_assignments
    ]
    paired_splits = {
        assignment.split
        for assignment in assignments
        if assignment.record.source_relative_path in {left.source_relative_path, right.source_relative_path}
    }
    assert len(paired_splits) == 1

    with pytest.raises(CriticalDataConflictError):
        _ensure_assignment_validation({
            "unique_source_paths": False,
            "no_sha256_across_splits": True,
            "source_groups_isolated": True,
            "allowed_source_only": True,
            "valid_destination_layout": True,
        })

from src.data.data_audit import sha256_file
from src.data.data_split import materialize_splits, validate_assignments, validate_staged_files
from src.data.final_dataset_resolution import _exclude_label_noise_records


def test_task_02e_excludes_label_noise_and_isolates_decided_visual_group(tmp_path: Path) -> None:
    metal_paths = [
        _write_training_image(tmp_path, "metal", f"metal_{index}_jpg", index + 1, (30 + index, 80, 150))
        for index in range(4)
    ]
    for index in range(8):
        _write_training_image(tmp_path, "cardboard", f"card_{index}_jpg", index + 20, (100, 20 + index, 40))
        _write_training_image(tmp_path, "paper", f"paper_{index}_jpg", index + 40, (40, 100, 20 + index))
    noise_left = _write_training_image(tmp_path, "cardboard", "noise_left_jpg", 80, (210, 30, 40))
    noise_right = _write_training_image(tmp_path, "paper", "noise_right_jpg", 81, (40, 30, 210))

    cleaning = run_cleaning(tmp_path)
    records_by_path = {record.source_path: record for record in cleaning.retained_records}
    noise_left_record = records_by_path[noise_left.resolve()]
    noise_right_record = records_by_path[noise_right.resolve()]
    excluded_paths = frozenset({
        noise_left_record.source_relative_path,
        noise_right_record.source_relative_path,
    })
    manual_rows = {
        "ND-002": {
            "user_decision": "label_noise_confirmed",
            "left_source_relative_path": noise_left_record.source_relative_path,
            "right_source_relative_path": noise_right_record.source_relative_path,
            "left_split": "test_internal",
            "right_split": "test_internal",
        }
    }
    active_records, excluded_records = _exclude_label_noise_records(
        cleaning.retained_records, manual_rows, excluded_paths
    )
    assert {record["source_relative_path"] for record in excluded_records} == excluded_paths
    assert all(record.source_relative_path not in excluded_paths for record in active_records)

    metal_records = [records_by_path[path.resolve()] for path in metal_paths]
    groups = ConfirmedVisualGroups(
        pairs_by_id={
            "ND-021": {
                "left_source_relative_path": metal_records[0].source_relative_path,
                "right_source_relative_path": metal_records[3].source_relative_path,
            },
            "ND-023": {
                "left_source_relative_path": metal_records[1].source_relative_path,
                "right_source_relative_path": metal_records[3].source_relative_path,
            },
            "ND-025": {
                "left_source_relative_path": metal_records[2].source_relative_path,
                "right_source_relative_path": metal_records[3].source_relative_path,
            },
        },
        confirmed_pair_ids=frozenset({"ND-021", "ND-023", "ND-025"}),
        ambiguous_pair_ids=frozenset(),
        false_positive_pair_ids=frozenset(),
    )
    source_hashes_before = {
        record.source_relative_path: sha256_file(record.source_path)
        for record in cleaning.retained_records
    }
    assignments = stratified_group_split(active_records, seed=42, confirmed_groups=groups)
    repeat_assignments = stratified_group_split(active_records, seed=42, confirmed_groups=groups)
    assert [(item.record.source_relative_path, item.split) for item in assignments] == [
        (item.record.source_relative_path, item.split) for item in repeat_assignments
    ]
    visual_splits = {
        item.split
        for item in assignments
        if item.record.source_relative_path in {record.source_relative_path for record in metal_records}
    }
    assert len(visual_splits) == 1

    validation = validate_assignments(assignments, tmp_path)
    assert validation["no_sha256_across_splits"]
    assert validation["source_groups_isolated"]
    assert validation["allowed_source_only"]
    output_root = tmp_path / "data" / "processed"
    manifest_rows = materialize_splits(assignments, tmp_path, output_root)
    assert validate_staged_files(manifest_rows, output_root)["all_staged_hashes_match"]
    assert all(row["source_relative_path"] not in excluded_paths for row in manifest_rows)
    assert sum(1 for _ in output_root.rglob("*.jpg")) == len(manifest_rows)
    assert source_hashes_before == {
        record.source_relative_path: sha256_file(record.source_path)
        for record in cleaning.retained_records
    }
