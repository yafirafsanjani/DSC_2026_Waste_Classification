from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.inference.competition import CompetitionDataError, CompetitionTestDataset, audit_competition_test, create_competition_dataloader, load_submission_template
from src.inference.predict import Prediction, load_cnn_best_checkpoint, predict_torch_classifier
from src.inference.submission import SubmissionValidationError, validate_submission_csv, write_submission_csv
from src.models import CNNFromScratch


def _make_project(tmp_path: Path, image_ids: tuple[str, ...] = ("010", "002")) -> Path:
    test_dir = tmp_path / "data" / "competition" / "test"
    test_dir.mkdir(parents=True)
    for index, image_id in enumerate(image_ids):
        suffix = ".png" if index == 0 else ".jpg"
        Image.fromarray(np.full((11 + index, 13, 3), 30 + index, dtype=np.uint8), mode="RGB").save(test_dir / f"{image_id}{suffix}")
    sample_path = tmp_path / "data" / "competition" / "SampleSubmission_DSC.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "waste_category"])
        for image_id in image_ids:
            writer.writerow([image_id, "template-value-must-not-be-used"])
    (tmp_path / "configs").mkdir()
    for source in (Path.cwd() / "configs").iterdir():
        if source.is_file():
            (tmp_path / "configs" / source.name).write_bytes(source.read_bytes())
    (tmp_path / "configs" / "models").mkdir()
    for source in (Path.cwd() / "configs" / "models").iterdir():
        (tmp_path / "configs" / "models" / source.name).write_bytes(source.read_bytes())
    return tmp_path


def test_competition_dataset_is_ordered_unlabeled_and_deterministic(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    report = audit_competition_test(project_root=root, expected_count=2)
    assert report.image_count == report.sample_row_count == 2
    dataset = CompetitionTestDataset(model_type="cnn", project_root=root, expected_count=2)
    first_tensor, first_id = dataset[0]
    repeated_tensor, repeated_id = dataset[0]
    assert [image_id for _, image_id in dataset] == ["010", "002"]
    assert first_id == repeated_id == "010"
    assert torch.equal(first_tensor, repeated_tensor)
    assert first_tensor.dtype == torch.float32 and first_tensor.shape == (3, 224, 224)
    assert not hasattr(dataset, "class_to_idx")


@pytest.mark.parametrize("forbidden_split", ("val", "test_internal"))
def test_competition_dataset_rejects_validation_or_internal_test_paths(tmp_path: Path, forbidden_split: str) -> None:
    root = _make_project(tmp_path)
    forbidden = root / "data" / "processed" / forbidden_split
    forbidden.mkdir(parents=True)
    with pytest.raises(CompetitionDataError, match="only"):
        CompetitionTestDataset(model_type="cnn", project_root=root, test_dir=forbidden)


def test_submission_replaces_template_values_and_preserves_leading_zero_order(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    template = load_submission_template(root / "data" / "competition" / "SampleSubmission_DSC.csv")
    predictions = [Prediction("010", "metal"), Prediction("002", "cardboard")]
    destination = root / "Final Dataset for Submission to Kaggle" / "submission_cnn_from_scratch.csv"
    other_model = destination.parent / "submission_mobilenetv3.csv"
    other_model.parent.mkdir(parents=True, exist_ok=True)
    other_model.write_text("unchanged", encoding="utf-8")
    write_submission_csv(template, predictions, destination, allowed_classes={"metal", "cardboard"})
    validate_submission_csv(destination, template, allowed_classes={"metal", "cardboard"})
    with destination.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["image_id"] for row in rows] == ["010", "002"]
    assert [row["waste_category"] for row in rows] == ["metal", "cardboard"]
    assert all(row["waste_category"] != "template-value-must-not-be-used" for row in rows)
    assert other_model.read_text(encoding="utf-8") == "unchanged"


def test_submission_rejects_prediction_order_mismatch(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    template = load_submission_template(root / "data" / "competition" / "SampleSubmission_DSC.csv")
    with pytest.raises(SubmissionValidationError, match="template order"):
        write_submission_csv(
            template, [Prediction("002", "metal"), Prediction("010", "cardboard")], root / "submission.csv",
            allowed_classes={"metal", "cardboard"},
        )


def test_checkpoint_loader_uses_artifact_mapping_and_does_not_mutate_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best.pt"
    model = CNNFromScratch(num_classes=2, dropout=0.40)
    class_mapping = {"alpha": 0, "beta": 1}
    torch.save(
        {"model_identifier": "cnn_from_scratch", "model_state_dict": model.state_dict(), "class_mapping": class_mapping,
         "training_config": {"dropout": 0.40}, "epoch": 4, "best_epoch": 4, "best_metric": 0.5}, checkpoint_path,
    )
    before_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    loaded = load_cnn_best_checkpoint(checkpoint_path=checkpoint_path, device="cpu", expected_class_mapping=class_mapping)
    assert loaded.class_mapping == class_mapping and loaded.index_to_class == ("alpha", "beta")
    assert loaded.model.training is False
    assert before_hash == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()


class _SyntheticInferenceDataset(Dataset[tuple[torch.Tensor, str]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        return torch.full((3, 4, 4), float(index)), f"00{index + 1}"


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3 * 4 * 4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs.flatten(1))


def test_torch_prediction_runs_in_eval_inference_mode_without_gradients() -> None:
    model = _TinyClassifier()
    model.train()
    predictions = predict_torch_classifier(model, DataLoader(_SyntheticInferenceDataset(), batch_size=2), index_to_class=("alpha", "beta"), device="cpu")
    assert model.training is False
    assert [prediction.image_id for prediction in predictions] == ["001", "002"]
    assert all(parameter.grad is None for parameter in model.parameters())


def test_competition_dataloader_is_sequential(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    loader = create_competition_dataloader(CompetitionTestDataset(model_type="cnn", project_root=root), batch_size=1, pin_memory=False)
    assert [image_ids[0] for _, image_ids in loader] == ["010", "002"]
