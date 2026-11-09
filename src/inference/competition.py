"""Leakage-safe access to official unlabeled competition test images."""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import build_transforms


SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


class CompetitionDataError(ValueError):
    """Raised when official test files or the submission template are unsafe."""


@dataclass(frozen=True)
class SubmissionTemplate:
    """Schema and ordered IDs only; sample category values are deliberately ignored."""

    columns: tuple[str, ...]
    image_id_column: str
    target_column: str
    image_ids: tuple[str, ...]


@dataclass(frozen=True)
class CompetitionAuditReport:
    test_directory: str
    image_count: int
    readable_image_count: int
    extensions: dict[str, int]
    formats: dict[str, int]
    color_modes: dict[str, int]
    unique_dimensions: int
    sample_submission_path: str
    sample_columns: tuple[str, ...]
    sample_row_count: int
    expected_count: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def competition_paths(project_root: Path | str | None = None) -> tuple[Path, Path]:
    root = Path(project_root or Path.cwd()).resolve()
    return root / "data" / "competition" / "test", root / "data" / "competition" / "SampleSubmission_DSC.csv"


def _official_path(project_root: Path | str | None, supplied: Path | str | None, expected: Path, name: str) -> Path:
    selected = Path(supplied).resolve() if supplied is not None else expected.resolve()
    if selected != expected.resolve():
        raise CompetitionDataError(
            f"Competition inference may use only {expected.relative_to(expected.parents[3]).as_posix()} for {name}."
        )
    return selected


def _test_dir(project_root: Path | str | None, supplied: Path | str | None) -> Path:
    expected, _ = competition_paths(project_root)
    selected = _official_path(project_root, supplied, expected, "test images")
    if not selected.is_dir():
        raise CompetitionDataError(f"Official competition test directory does not exist: {selected}")
    return selected


def _sample_path(project_root: Path | str | None, supplied: Path | str | None) -> Path:
    _, expected = competition_paths(project_root)
    selected = _official_path(project_root, supplied, expected, "sample submission")
    if not selected.is_file():
        raise CompetitionDataError(f"Official sample submission does not exist: {selected}")
    return selected


def load_submission_template(
    sample_submission_path: Path | str,
    *,
    image_id_column: str = "image_id",
    target_column: str = "waste_category",
) -> SubmissionTemplate:
    """Read header and IDs while never consuming sample category values as labels."""
    with Path(sample_submission_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            columns = tuple(next(reader))
        except StopIteration as error:
            raise CompetitionDataError("Sample submission is empty.") from error
        if set(columns) != {image_id_column, target_column} or len(columns) != 2:
            raise CompetitionDataError(f"Sample submission columns must be {image_id_column!r} and {target_column!r}; found {list(columns)!r}.")
        image_id_index = columns.index(image_id_column)
        image_ids: list[str] = []
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(columns):
                raise CompetitionDataError(f"Sample submission row {row_number} has an invalid field count.")
            if not row[image_id_index]:
                raise CompetitionDataError(f"Sample submission row {row_number} has an empty image_id.")
            image_ids.append(row[image_id_index])
    duplicates = [image_id for image_id, count in Counter(image_ids).items() if count > 1]
    if duplicates:
        raise CompetitionDataError(f"Sample submission has duplicate image IDs: {sorted(duplicates)[:5]!r}.")
    return SubmissionTemplate(columns, image_id_column, target_column, tuple(image_ids))


def _index_images(test_dir: Path, inspect: bool) -> tuple[dict[str, Path], Counter[str], Counter[str], Counter[str], set[tuple[int, int]]]:
    directories = [path.name for path in test_dir.iterdir() if path.is_dir()]
    if directories:
        raise CompetitionDataError(f"Official test directory must be flat and unlabeled; found subdirectories: {sorted(directories)[:5]!r}.")
    files = sorted(path for path in test_dir.iterdir() if path.is_file())
    unsupported = [path.name for path in files if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES]
    if unsupported:
        raise CompetitionDataError(f"Unsupported files in official test directory: {unsupported[:5]!r}.")
    image_by_id: dict[str, Path] = {}
    extensions: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    dimensions: set[tuple[int, int]] = set()
    for path in files:
        if not path.stem or path.stem in image_by_id:
            raise CompetitionDataError(f"Duplicate or empty image ID stem: {path.name!r}.")
        image_by_id[path.stem] = path
        extensions[path.suffix.lower()] += 1
        if inspect:
            try:
                with Image.open(path) as image:
                    image.load()
                    formats[str(image.format)] += 1
                    modes[str(image.mode)] += 1
                    dimensions.add(image.size)
            except (OSError, UnidentifiedImageError, ValueError) as error:
                raise CompetitionDataError(f"Official test image is unreadable: {path.name}: {error}") from error
    return image_by_id, extensions, formats, modes, dimensions


def audit_competition_test(
    *,
    project_root: Path | str | None = None,
    test_dir: Path | str | None = None,
    sample_submission_path: Path | str | None = None,
    expected_count: int | None = None,
) -> CompetitionAuditReport:
    """Fail closed unless every official image and template ID match exactly."""
    official_test_dir = _test_dir(project_root, test_dir)
    official_sample_path = _sample_path(project_root, sample_submission_path)
    template = load_submission_template(official_sample_path)
    image_by_id, extensions, formats, modes, dimensions = _index_images(official_test_dir, inspect=True)
    missing_images = sorted(set(template.image_ids) - set(image_by_id))
    extra_images = sorted(set(image_by_id) - set(template.image_ids))
    if missing_images or extra_images:
        raise CompetitionDataError(f"Official test/template ID mismatch; missing images={missing_images[:5]!r}, extra images={extra_images[:5]!r}.")
    if expected_count is not None and (len(image_by_id) != expected_count or len(template.image_ids) != expected_count):
        raise CompetitionDataError(f"Expected {expected_count} test images and rows; found {len(image_by_id)} images and {len(template.image_ids)} rows.")
    return CompetitionAuditReport(
        test_directory=str(official_test_dir), image_count=len(image_by_id), readable_image_count=len(image_by_id),
        extensions=dict(sorted(extensions.items())), formats=dict(sorted(formats.items())), color_modes=dict(sorted(modes.items())),
        unique_dimensions=len(dimensions), sample_submission_path=str(official_sample_path), sample_columns=template.columns,
        sample_row_count=len(template.image_ids), expected_count=expected_count,
    )


class CompetitionTestDataset(Dataset[tuple[torch.Tensor, str]]):
    """Ordered unlabeled dataset backed exclusively by official competition images."""

    def __init__(
        self, *, model_type: str, project_root: Path | str | None = None, test_dir: Path | str | None = None,
        sample_submission_path: Path | str | None = None, transform: Callable[[Path], torch.Tensor] | None = None,
        expected_count: int | None = None,
    ) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.test_dir = _test_dir(self.project_root, test_dir)
        self.sample_submission_path = _sample_path(self.project_root, sample_submission_path)
        self.template = load_submission_template(self.sample_submission_path)
        image_by_id, _, _, _, _ = _index_images(self.test_dir, inspect=False)
        missing_images = sorted(set(self.template.image_ids) - set(image_by_id))
        extra_images = sorted(set(image_by_id) - set(self.template.image_ids))
        if missing_images or extra_images:
            raise CompetitionDataError(f"Official test/template ID mismatch; missing images={missing_images[:5]!r}, extra images={extra_images[:5]!r}.")
        if expected_count is not None and len(self.template.image_ids) != expected_count:
            raise CompetitionDataError(f"Expected {expected_count} test IDs; found {len(self.template.image_ids)}.")
        self.records = tuple((image_id, image_by_id[image_id]) for image_id in self.template.image_ids)
        self.transform = transform or build_transforms(model_type, "inference", project_root=self.project_root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_id, image_path = self.records[index]
        try:
            return self.transform(image_path), image_id
        except Exception as error:
            raise CompetitionDataError(f"Unable to preprocess official image {image_id!r}: {error}") from error


def create_competition_dataloader(
    dataset: CompetitionTestDataset, *, batch_size: int, num_workers: int = 0, pin_memory: bool | None = None,
) -> DataLoader[tuple[torch.Tensor, str]]:
    """Build a sequential inference loader; shuffling is never permitted."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory)
