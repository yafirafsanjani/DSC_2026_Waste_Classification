"""Prepare the final read-only visual review for selected V1 near-duplicate pairs."""

from __future__ import annotations

import argparse
import csv
import logging
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.data.data_audit import hamming_distance

LOGGER = logging.getLogger(__name__)

TARGET_PAIR_IDS = ("ND-021", "ND-023", "ND-025", "ND-002")
SOURCE_PREFIX = "data/extracted/v1/train/"
REVIEW_COLUMNS = (
    "pair_id",
    "review_purpose",
    "left_filename",
    "left_source_relative_path",
    "left_class",
    "left_split",
    "left_source_identity",
    "left_sha256",
    "left_perceptual_hash",
    "right_filename",
    "right_source_relative_path",
    "right_class",
    "right_split",
    "right_source_identity",
    "right_sha256",
    "right_perceptual_hash",
    "reported_perceptual_hash_distance",
    "calculated_perceptual_hash_distance",
    "same_sha256",
    "same_source_identity",
    "same_class",
    "split_combination",
    "existing_relationship_assessment",
    "existing_leakage_risk",
    "confirmed_visual_group_id",
    "confirmed_visual_group_status",
    "prior_review_status",
    "user_decision",
    "user_decision_options",
    "user_rationale",
    "reviewer_name",
    "reviewed_at",
    "follow_up_action",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read UTF-8 CSV metadata without changing it."""

    with path.open(encoding="utf-8-sig", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Write a deterministic manual-review template."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _require_v1_train_path(relative_path: str) -> None:
    normalized_path = relative_path.replace("\\", "/")
    if not normalized_path.startswith(SOURCE_PREFIX):
        raise ValueError(f"Manual review only accepts V1/train paths: {relative_path}")


def _load_target_rows(project_root: Path) -> list[dict[str, str]]:
    """Load requested pairs and verify their current split-manifest mappings."""

    metadata_dir = project_root / "data" / "metadata"
    review_rows = _read_csv(metadata_dir / "near_duplicate_split_review.csv")
    manifest_rows = _read_csv(metadata_dir / "split_manifest.csv")
    manifest_by_source = {row["source_relative_path"]: row for row in manifest_rows}
    review_by_id = {row["pair_id"]: row for row in review_rows}

    missing_ids = [pair_id for pair_id in TARGET_PAIR_IDS if pair_id not in review_by_id]
    if missing_ids:
        raise ValueError(f"Requested pair IDs are absent from the near-duplicate review: {missing_ids}")

    selected_rows: list[dict[str, str]] = []
    manifest_field_by_review_suffix = {
        "split": "split",
        "class": "class_name",
        "sha256": "sha256",
        "perceptual_hash": "perceptual_hash",
        "source_identity": "source_identity",
    }
    for pair_id in TARGET_PAIR_IDS:
        row = review_by_id[pair_id]
        if row.get("manifest_mapping_status") != "mapped":
            raise ValueError(f"Pair {pair_id} is not mapped to the current split manifest.")
        for side in ("left", "right"):
            source_path = row[f"{side}_source_relative_path"]
            _require_v1_train_path(source_path)
            manifest_record = manifest_by_source.get(source_path)
            if manifest_record is None:
                raise ValueError(f"Pair {pair_id} source path is missing from the split manifest: {source_path}")
            for review_suffix, manifest_field in manifest_field_by_review_suffix.items():
                if manifest_record[manifest_field] != row[f"{side}_{review_suffix}"]:
                    raise ValueError(
                        f"Pair {pair_id} {side} metadata disagrees with split manifest for {manifest_field}."
                    )
        calculated_distance = hamming_distance(row["left_perceptual_hash"], row["right_perceptual_hash"])
        if calculated_distance != int(row["calculated_perceptual_hash_distance"]):
            raise ValueError(f"Pair {pair_id} perceptual hash distance is inconsistent.")
        selected_rows.append(row)

    return selected_rows


def _visual_group_lookup(project_root: Path) -> dict[str, list[dict[str, str]]]:
    """Index confirmed group rows by pair ID without inferring new relations."""

    groups = _read_csv(project_root / "data" / "metadata" / "near_duplicate_groups.csv")
    pairs_to_groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for group_row in groups:
        serialized_pair_ids = group_row["confirmed_pair_ids"].strip().strip("[]")
        for pair_id in (
            value.strip().strip('"') for value in serialized_pair_ids.split(",") if value.strip()
        ):
            pairs_to_groups[pair_id].append(group_row)
    return dict(pairs_to_groups)


def build_manual_review_rows(project_root: Path | str) -> list[dict[str, str]]:
    """Build a blank-decision template for the four requested evidence pairs."""

    root = Path(project_root).resolve()
    selected_rows = _load_target_rows(root)
    groups_by_pair = _visual_group_lookup(root)
    output_rows: list[dict[str, str]] = []

    for row in selected_rows:
        pair_id = row["pair_id"]
        group_rows = groups_by_pair.get(pair_id, [])
        group_ids = sorted({group_row["visual_group_id"] for group_row in group_rows})
        group_statuses = sorted({group_row["group_status"] for group_row in group_rows})
        label_noise_review = pair_id == "ND-002"
        output_rows.append({
            "pair_id": pair_id,
            "review_purpose": "potential_label_noise" if label_noise_review else "ambiguous_cross_split_similarity",
            "left_filename": Path(row["left_source_relative_path"]).name,
            "left_source_relative_path": row["left_source_relative_path"],
            "left_class": row["left_class"],
            "left_split": row["left_split"],
            "left_source_identity": row["left_source_identity"],
            "left_sha256": row["left_sha256"],
            "left_perceptual_hash": row["left_perceptual_hash"],
            "right_filename": Path(row["right_source_relative_path"]).name,
            "right_source_relative_path": row["right_source_relative_path"],
            "right_class": row["right_class"],
            "right_split": row["right_split"],
            "right_source_identity": row["right_source_identity"],
            "right_sha256": row["right_sha256"],
            "right_perceptual_hash": row["right_perceptual_hash"],
            "reported_perceptual_hash_distance": row["reported_perceptual_hash_distance"],
            "calculated_perceptual_hash_distance": row["calculated_perceptual_hash_distance"],
            "same_sha256": row["same_sha256"],
            "same_source_identity": row["same_source_identity"],
            "same_class": row["same_class"],
            "split_combination": row["split_combination"],
            "existing_relationship_assessment": row["relationship_assessment"],
            "existing_leakage_risk": row["leakage_risk"],
            "confirmed_visual_group_id": ";".join(group_ids),
            "confirmed_visual_group_status": ";".join(group_statuses),
            "prior_review_status": "label_noise_review_required" if label_noise_review else "manual_leakage_decision_required",
            "user_decision": "",
            "user_decision_options": (
                "label_correct|label_noise_confirmed|uncertain"
                if label_noise_review
                else "same_source_or_derivative|different_source|uncertain"
            ),
            "user_rationale": "",
            "reviewer_name": "",
            "reviewed_at": "",
            "follow_up_action": (
                "Do not modify labels automatically; record the decision before any cleaning action."
                if label_noise_review
                else "Do not resplit automatically; record the decision before any split change."
            ),
        })
    return output_rows


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Use a readable installed font while retaining a Pillow fallback."""

    font_names = (
        ("C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf")
        if bold
        else ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf")
    )
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    max_characters: int,
    fill: str = "#18212b",
) -> int:
    """Draw a wrapped label and return the y-coordinate after the block."""

    x_position, y_position = position
    line_height = int(getattr(font, "size", 14) * 1.28)
    for source_line in text.splitlines():
        for line in textwrap.wrap(source_line, width=max_characters) or [""]:
            draw.text((x_position, y_position), line, fill=fill, font=font)
            y_position += line_height
    return y_position


def _draw_image_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image_path: Path,
    origin: tuple[int, int],
    panel_size: tuple[int, int],
    title: str,
    details: str,
    body_font: ImageFont.ImageFont,
    title_font: ImageFont.ImageFont,
) -> None:
    """Render a read-only source image with its split and label metadata."""

    x_position, y_position = origin
    panel_width, panel_height = panel_size
    draw.rounded_rectangle(
        (x_position, y_position, x_position + panel_width, y_position + panel_height),
        radius=24,
        fill="#f7f4ed",
        outline="#60758a",
        width=4,
    )
    text_y = _draw_wrapped_text(draw, (x_position + 32, y_position + 28), title, title_font, 45)
    text_y = _draw_wrapped_text(draw, (x_position + 32, text_y + 8), details, body_font, 56)
    image_box = (x_position + 32, text_y + 22, x_position + panel_width - 32, y_position + panel_height - 32)
    try:
        with Image.open(image_path) as opened_image:
            display_image = ImageOps.exif_transpose(opened_image).convert("RGB")
            display_image = ImageOps.contain(
                display_image,
                (image_box[2] - image_box[0], image_box[3] - image_box[1]),
                Image.Resampling.LANCZOS,
            )
            paste_x = image_box[0] + ((image_box[2] - image_box[0]) - display_image.width) // 2
            paste_y = image_box[1] + ((image_box[3] - image_box[1]) - display_image.height) // 2
            canvas.paste(display_image, (paste_x, paste_y))
    except OSError as error:
        _draw_wrapped_text(
            draw,
            (image_box[0] + 20, image_box[1] + 20),
            f"Could not open image: {error}",
            body_font,
            48,
            "#a30000",
        )


def create_final_contact_sheet(project_root: Path | str, rows: Sequence[dict[str, str]]) -> Path:
    """Create one high-resolution sheet from source V1/train images only."""

    root = Path(project_root).resolve()
    output_path = root / "outputs" / "figures" / "final_manual_review.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas_width = 3600
    page_margin = 100
    column_gap = 60
    header_height = 250
    row_height = 1400
    row_gap = 60
    panel_width = (canvas_width - (2 * page_margin) - column_gap) // 2
    canvas_height = header_height + page_margin + (len(rows) * row_height) + ((len(rows) - 1) * row_gap) + page_margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), "#e8eef2")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(48, bold=True)
    panel_title_font = _load_font(34, bold=True)
    body_font = _load_font(28)

    draw.text((page_margin, 50), "DSC 2026 - Final Manual Review", fill="#102a43", font=title_font)
    draw.text(
        (page_margin, 120),
        "Read-only evidence sheet. No split, label, or source image has been changed.",
        fill="#334e68",
        font=body_font,
    )

    for row_index, row in enumerate(rows):
        row_top = header_height + (row_index * (row_height + row_gap))
        header = (
            f"{row['pair_id']} | {row['review_purpose']} | pHash distance="
            f"{row['calculated_perceptual_hash_distance']} | {row['split_combination']}"
        )
        draw.text((page_margin, row_top), header, fill="#102a43", font=panel_title_font)
        panel_top = row_top + 60
        panel_height = row_height - 60
        left_details = (
            f"split={row['left_split']} | label={row['left_class']} | identity={row['left_source_identity']}\n"
            f"file={row['left_filename']}\nsha256={row['left_sha256'][:16]}..."
        )
        right_details = (
            f"split={row['right_split']} | label={row['right_class']} | identity={row['right_source_identity']}\n"
            f"file={row['right_filename']}\nsha256={row['right_sha256'][:16]}..."
        )
        _draw_image_panel(
            canvas,
            draw,
            root / row["left_source_relative_path"],
            (page_margin, panel_top),
            (panel_width, panel_height),
            "LEFT SOURCE IMAGE",
            left_details,
            body_font,
            panel_title_font,
        )
        _draw_image_panel(
            canvas,
            draw,
            root / row["right_source_relative_path"],
            (page_margin + panel_width + column_gap, panel_top),
            (panel_width, panel_height),
            "RIGHT SOURCE IMAGE",
            right_details,
            body_font,
            panel_title_font,
        )

    canvas.save(output_path, format="PNG", optimize=True)
    return output_path


def run_final_manual_review(project_root: Path | str) -> dict[str, Any]:
    """Generate the decision template and high-resolution contact sheet."""

    root = Path(project_root).resolve()
    rows = build_manual_review_rows(root)
    report_path = root / "data" / "metadata" / "final_manual_review.csv"
    _write_csv(report_path, rows)
    contact_sheet_path = create_final_contact_sheet(root, rows)
    LOGGER.info("Prepared %d final manual-review rows.", len(rows))
    return {
        "pair_ids": [row["pair_id"] for row in rows],
        "report_path": report_path.relative_to(root).as_posix(),
        "contact_sheet_path": contact_sheet_path.relative_to(root).as_posix(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final V1 near-duplicate manual-review artifacts.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_final_manual_review(arguments.project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
