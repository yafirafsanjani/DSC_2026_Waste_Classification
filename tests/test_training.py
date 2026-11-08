from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.training_manager import TrainingManager
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.visualization import plot_learning_curves
from src.training.train import train_one_epoch
from src.training.validate import validate_one_epoch


def _loader() -> DataLoader:
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    targets = torch.tensor([0, 1, 0, 1])
    return DataLoader(TensorDataset(inputs, targets), batch_size=3, shuffle=False)


def test_train_accumulation_and_validation_metrics() -> None:
    torch.manual_seed(42)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    train_metrics, _ = train_one_epoch(model, _loader(), criterion, optimizer, gradient_accumulation_steps=2, show_progress=False)
    assert train_metrics["sample_count"] == 4 and torch.isfinite(torch.tensor(train_metrics["loss"]))
    validation = validate_one_epoch(model, _loader(), criterion, {"alpha": 0, "beta": 1})
    assert validation["sample_count"] == 4
    assert {"accuracy", "macro_precision", "macro_recall", "macro_f1", "loss"}.issubset(validation)


def test_metrics_confusion_matrix_has_class_mapping_order() -> None:
    metrics = compute_classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], {"alpha": 0, "beta": 1})
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["per_class"]["alpha"]["support"] == 2


def test_manager_serializes_actual_synthetic_history_and_plot(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = nn.Linear(2, 2)
    manager = TrainingManager(model, torch.optim.SGD(model.parameters(), lr=0.1), nn.CrossEntropyLoss(),
                              {"alpha": 0, "beta": 1}, "tiny", "synthetic", project_root=tmp_path)
    history = manager.fit(_loader(), _loader(), max_epochs=1, show_progress=False)
    assert len(history) == 1 and manager.metrics_path.is_file() and manager.log_path.is_file()
    assert manager.checkpoints.best_path.is_file() and history[0]["checkpoint_best_saved"]
    output = plot_learning_curves(manager.metrics_path, tmp_path / "curve.png")
    assert output.is_file()


def test_nonfinite_loss_is_rejected() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(FloatingPointError):
        train_one_epoch(model, _loader(), lambda outputs, targets: torch.tensor(float("nan"), requires_grad=True), optimizer, show_progress=False)
