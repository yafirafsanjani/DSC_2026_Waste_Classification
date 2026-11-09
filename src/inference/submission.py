"""Kaggle-compatible submission generation with strict prediction alignment checks."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from .competition import CompetitionDataError, SubmissionTemplate, load_submission_template
from .predict import Prediction


class SubmissionValidationError(ValueError):
    """Raised when a submission cannot be proven aligned to the official template."""


def validate_predictions(
    template: SubmissionTemplate, predictions: Sequence[Prediction], *, allowed_classes: Iterable[str],
) -> None:
    """Verify count, order, IDs, and categories before any CSV is written."""
    allowed = frozenset(allowed_classes)
    if not allowed:
        raise SubmissionValidationError("At least one allowed class is required.")
    if len(predictions) != len(template.image_ids):
        raise SubmissionValidationError(f"Prediction count {len(predictions)} does not match template count {len(template.image_ids)}.")
    prediction_ids = [prediction.image_id for prediction in predictions]
    if len(set(prediction_ids)) != len(prediction_ids):
        raise SubmissionValidationError("Predictions contain duplicate image IDs.")
    if prediction_ids != list(template.image_ids):
        raise SubmissionValidationError("Prediction image IDs do not exactly match the template order.")
    invalid = sorted({prediction.waste_category for prediction in predictions if not prediction.waste_category or prediction.waste_category not in allowed})
    if invalid:
        raise SubmissionValidationError(f"Predictions contain invalid waste categories: {invalid!r}.")


def write_submission_csv(
    template: SubmissionTemplate, predictions: Sequence[Prediction], output_path: Path | str, *,
    allowed_classes: Iterable[str], overwrite: bool = False,
) -> Path:
    """Write model predictions only; sample category values are never copied."""
    validate_predictions(template, predictions, allowed_classes=allowed_classes)
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Submission already exists and overwrite is disabled: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(template.columns), lineterminator="\n")
        writer.writeheader()
        for prediction in predictions:
            writer.writerow({template.image_id_column: prediction.image_id, template.target_column: prediction.waste_category})
    return destination


def validate_submission_csv(
    submission_path: Path | str, template: SubmissionTemplate, *, allowed_classes: Iterable[str],
) -> None:
    """Re-read an output CSV as strings and enforce the same invariants."""
    path = Path(submission_path)
    if not path.is_file():
        raise SubmissionValidationError(f"Submission CSV does not exist: {path}")
    try:
        output_template = load_submission_template(path, image_id_column=template.image_id_column, target_column=template.target_column)
    except CompetitionDataError as error:
        raise SubmissionValidationError(str(error)) from error
    if output_template.columns != template.columns:
        raise SubmissionValidationError("Submission columns do not match the official sample submission order.")
    if output_template.image_ids != template.image_ids:
        raise SubmissionValidationError("Submission image IDs do not match the official sample submission order.")
    allowed = frozenset(allowed_classes)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        categories = [row.get(template.target_column, "") for row in csv.DictReader(handle)]
    if len(categories) != len(template.image_ids) or any(not category for category in categories):
        raise SubmissionValidationError("Submission has an invalid row count or an empty waste category.")
    invalid = sorted(set(categories) - allowed)
    if invalid:
        raise SubmissionValidationError(f"Submission contains invalid waste categories: {invalid!r}.")
