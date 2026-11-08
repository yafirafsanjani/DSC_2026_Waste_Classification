"""Review TASK 02 near-duplicate candidates against the finalized V1 split."""

from __future__ import annotations

import argparse
import csv
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

from src.data.data_audit import hamming_distance, write_csv

LOGGER = logging.getLogger(__name__)

SPLIT_ORDER = {"train": 0, "val": 1, "test_internal": 2}
SPLIT_DISPLAY = {"train": "Train", "val": "Validation", "test_internal": "Internal Test"}
COMBINATION_NAMES = {
    ("train", "train"): "TrainTrain",
    ("train", "val"): "TrainValidation",
    ("train", "test_internal"): "TrainInternal Test",
    ("val", "val"): "ValidationValidation",
    ("val", "test_internal"): "ValidationInternal Test",
    ("test_internal", "test_internal"): "Internal TestInternal Test",
}


@dataclass(frozen=True)
class ManifestRecord:
    """A verified V1/train source record from the finalized split manifest."""

    source_relative_path: str
    processed_relative_path: str
    split: str
    class_name: str
    source_identity: str | None
    sha256: str
    perceptual_hash: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def _bool_from_csv(value: str) -> bool:
    return value.strip().lower() == "true"


def split_combination(left_split: str, right_split: str) -> str:
    """Return the stable, unordered split-combination label used in reports."""

    if left_split not in SPLIT_ORDER or right_split not in SPLIT_ORDER:
        raise ValueError(f"Unknown split combination: {left_split}, {right_split}")
    ordered = tuple(sorted((left_split, right_split), key=lambda split: SPLIT_ORDER[split]))
    return COMBINATION_NAMES[ordered]


def _relationship_assessment(
    same_sha256: bool, same_source_identity: bool, perceptual_distance: int
) -> tuple[str, str]:
    """Describe evidence strength without treating visual similarity as proof."""

    if same_sha256:
        return "exact_same_file", "high"
    if same_source_identity:
        return "same_roboflow_source_identity", "high"
    if perceptual_distance == 0:
        return "very_high_visual_similarity_distinct_source_metadata", "review"
    if perceptual_distance <= 4:
        return "possible_augmentation_or_same_scene_distinct_source_metadata", "review"
    return "visually_similar_distinct_source_metadata", "review"


def load_v1_split_manifest(project_root: Path | str) -> dict[str, ManifestRecord]:
    """Load only manifest entries whose provenance is the permitted V1/train source."""

    root = Path(project_root).resolve()
    manifest_path = root / "data" / "metadata" / "split_manifest.csv"
    by_source: dict[str, ManifestRecord] = {}
    for row in _read_csv(manifest_path):
        source_path = row["source_relative_path"]
        if not source_path.startswith("data/extracted/v1/train/"):
            raise ValueError(f"Disallowed provenance in split manifest: {source_path}")
        processed_path = row["processed_relative_path"]
        if not processed_path.startswith("data/processed/"):
            raise ValueError(f"Invalid processed path in split manifest: {processed_path}")
        if source_path in by_source:
            raise ValueError(f"Duplicate source path in split manifest: {source_path}")
        by_source[source_path] = ManifestRecord(
            source_relative_path=source_path,
            processed_relative_path=processed_path,
            split=row["split"],
            class_name=row["class_name"],
            source_identity=row["source_identity"] or None,
            sha256=row["sha256"],
            perceptual_hash=row["perceptual_hash"],
        )
    return by_source


def load_near_duplicate_candidates(project_root: Path | str) -> list[dict[str, str]]:
    """Load only near-duplicate rows written by the V1/train cleaning report."""

    root = Path(project_root).resolve()
    report_path = root / "data" / "metadata" / "cleaning_report.csv"
    candidates = [
        row for row in _read_csv(report_path) if row.get("record_type") == "near_duplicate"
    ]
    for row in candidates:
        for key in ("left_path", "right_path"):
            if not row[key].startswith("data/extracted/v1/train/"):
                raise ValueError(f"Disallowed near-duplicate candidate path: {row[key]}")
    return candidates


def build_review_rows(
    candidates: Iterable[dict[str, str]], manifest: dict[str, ManifestRecord]
) -> list[dict[str, Any]]:
    """Join V1 near-duplicate candidates to finalized split assignments."""

    rows: list[dict[str, Any]] = []
    for pair_index, candidate in enumerate(candidates, start=1):
        left_path = candidate["left_path"]
        right_path = candidate["right_path"]
        left = manifest.get(left_path)
        right = manifest.get(right_path)
        base_row: dict[str, Any] = {
            "pair_id": f"ND-{pair_index:03d}",
            "left_source_relative_path": left_path,
            "right_source_relative_path": right_path,
            "reported_perceptual_hash_distance": int(candidate["hamming_distance"]),
        }
        if left is None or right is None:
            base_row.update({
                "manifest_mapping_status": "missing_source_in_split_manifest",
                "split_combination": "Unmapped",
                "manual_review_required": True,
                "leakage_risk": "high",
                "relationship_assessment": "unmapped_candidate_requires_investigation",
            })
            rows.append(base_row)
            continue

        calculated_distance = hamming_distance(left.perceptual_hash, right.perceptual_hash)
        same_sha256 = left.sha256 == right.sha256
        same_source_identity = (
            left.source_identity is not None and left.source_identity == right.source_identity
        )
        assessment, evidence_risk = _relationship_assessment(
            same_sha256, same_source_identity, calculated_distance
        )
        cross_split = left.split != right.split
        leakage_risk = evidence_risk if cross_split else "none_within_same_split"
        base_row.update({
            "manifest_mapping_status": "mapped",
            "left_processed_relative_path": left.processed_relative_path,
            "right_processed_relative_path": right.processed_relative_path,
            "left_split": left.split,
            "right_split": right.split,
            "left_class": left.class_name,
            "right_class": right.class_name,
            "left_source_identity": left.source_identity,
            "right_source_identity": right.source_identity,
            "left_sha256": left.sha256,
            "right_sha256": right.sha256,
            "left_perceptual_hash": left.perceptual_hash,
            "right_perceptual_hash": right.perceptual_hash,
            "calculated_perceptual_hash_distance": calculated_distance,
            "perceptual_hash_distance_verified": calculated_distance == int(candidate["hamming_distance"]),
            "same_sha256": same_sha256,
            "same_source_identity": same_source_identity,
            "same_class": left.class_name == right.class_name,
            "split_combination": split_combination(left.split, right.split),
            "cross_split": cross_split,
            "relationship_assessment": assessment,
            "leakage_risk": leakage_risk,
            "manual_review_required": cross_split,
            "review_note": (
                "Cross-split perceptual-hash candidate; inspect the contact sheet before changing splits."
                if cross_split
                else "Within one split; not cross-split leakage evidence."
            ),
        })
        rows.append(base_row)
    return rows


def _clean_previous_contact_sheets(figures_dir: Path) -> None:
    """Remove only contact-sheet files generated by this reviewer on a rerun."""

    for path in [figures_dir / "near_duplicate_contact_sheet.png", *figures_dir.glob("near_duplicate_contact_sheet_*.png")]:
        if path.exists():
            path.unlink()


def _show_image(axis: Any, image_path: Path, title: str) -> None:
    try:
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(title, fontsize=8, loc="left", wrap=True)
    except OSError as error:
        axis.text(0.5, 0.5, f"Could not open image\n{error}", ha="center", va="center")
        axis.set_title(title, fontsize=8, loc="left", wrap=True)
    axis.axis("off")


def create_contact_sheets(
    review_rows: Sequence[dict[str, Any]], project_root: Path | str, pairs_per_sheet: int = 6
) -> list[Path]:
    """Render V1 processed-image pairs for visual inspection without changing images."""

    if pairs_per_sheet < 1:
        raise ValueError("pairs_per_sheet must be at least one.")
    root = Path(project_root).resolve()
    figures_dir = root / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _clean_previous_contact_sheets(figures_dir)

    mapped_rows = [row for row in review_rows if row.get("manifest_mapping_status") == "mapped"]
    if not mapped_rows:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_paths: list[Path] = []
    page_count = math.ceil(len(mapped_rows) / pairs_per_sheet)
    for page_index in range(page_count):
        page_rows = mapped_rows[page_index * pairs_per_sheet:(page_index + 1) * pairs_per_sheet]
        figure, axes = plt.subplots(len(page_rows), 2, figsize=(14, 3.7 * len(page_rows)))
        if len(page_rows) == 1:
            axes = [axes]
        for axes_row, row in zip(axes, page_rows):
            distance = row["calculated_perceptual_hash_distance"]
            header = (
                f"{row['pair_id']} | {row['split_combination']} | pHash distance={distance} | "
                f"risk={row['leakage_risk']}"
            )
            _show_image(
                axes_row[0], root / row["left_processed_relative_path"],
                f"{header}\nLEFT: {row['left_split']} / {row['left_class']}\n{Path(row['left_source_relative_path']).name}",
            )
            _show_image(
                axes_row[1], root / row["right_processed_relative_path"],
                f"RIGHT: {row['right_split']} / {row['right_class']}\n{Path(row['right_source_relative_path']).name}",
            )
        figure.tight_layout()
        output_path = (
            figures_dir / "near_duplicate_contact_sheet.png"
            if page_index == 0
            else figures_dir / f"near_duplicate_contact_sheet_{page_index + 1:02d}.png"
        )
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def run_near_duplicate_review(
    project_root: Path | str, pairs_per_sheet: int = 6
) -> dict[str, Any]:
    """Create the CSV review and contact sheets for the final V1 split only."""

    root = Path(project_root).resolve()
    manifest = load_v1_split_manifest(root)
    candidates = load_near_duplicate_candidates(root)
    review_rows = build_review_rows(candidates, manifest)
    write_csv(root / "data" / "metadata" / "near_duplicate_split_review.csv", review_rows)
    sheets = create_contact_sheets(review_rows, root, pairs_per_sheet)

    combinations = Counter(row["split_combination"] for row in review_rows)
    cross_split_rows = [row for row in review_rows if row.get("cross_split")]
    summary = {
        "candidate_pairs": len(review_rows),
        "mapped_pairs": sum(row.get("manifest_mapping_status") == "mapped" for row in review_rows),
        "cross_split_pairs": len(cross_split_rows),
        "cross_split_risk_counts": dict(Counter(row["leakage_risk"] for row in cross_split_rows)),
        "split_combination_counts": {name: combinations.get(name, 0) for name in COMBINATION_NAMES.values()},
        "contact_sheets": [path.relative_to(root).as_posix() for path in sheets],
    }
    LOGGER.info(
        "Reviewed %d candidates: %d cross-split pairs and %d contact sheet(s).",
        summary["candidate_pairs"], summary["cross_split_pairs"], len(sheets),
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review V1/train near-duplicate candidates against final splits.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--pairs-per-sheet", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_near_duplicate_review(arguments.project_root, arguments.pairs_per_sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
