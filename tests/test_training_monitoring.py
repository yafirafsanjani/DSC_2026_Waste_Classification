from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.training_manager import TrainingManager
from src.evaluation.training_monitoring import history_review, load_training_history, persist_training_history
from src.training.train import train_one_epoch


def _loader() -> DataLoader:
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    targets = torch.tensor([0, 1, 0, 1])
    return DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)


def test_batch_progress_reports_loss_accuracy_and_learning_rate(monkeypatch) -> None:
    updates: list[dict[str, str]] = []
    progress_options: list[dict[str, object]] = []

    class FakeProgress:
        def __init__(self, iterable, **kwargs) -> None:
            self.iterable = iterable
            self.kwargs = kwargs
            progress_options.append(kwargs)

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs) -> None:
            updates.append(kwargs)

    monkeypatch.setattr("src.training.train.tqdm", FakeProgress)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_one_epoch(model, _loader(), nn.CrossEntropyLoss(), optimizer, show_progress=True, epoch=2, max_epochs=5)
    assert updates and {"loss", "acc", "lr"}.issubset(updates[-1])
    assert progress_options[0]["desc"] == "Epoch 3/5 train"


def test_manager_persists_atomic_json_csv_jsonl_and_epoch_summary(tmp_path: Path, capsys) -> None:
    model = nn.Linear(2, 2)
    manager = TrainingManager(
        model, torch.optim.SGD(model.parameters(), lr=0.1), nn.CrossEntropyLoss(), {"alpha": 0, "beta": 1},
        "tiny", "monitoring_synthetic", project_root=tmp_path,
    )
    history = manager.fit(_loader(), _loader(), max_epochs=1, show_progress=False)
    record = history[0]
    assert manager.metrics_path.is_file() and manager.metrics_csv_path.is_file() and manager.log_path.is_file()
    assert {"experiment_id", "learning_rate", "learning_rate_changed", "checkpoint_status", "early_stopping_counter", "best_validation_loss"}.issubset(record)
    assert record["checkpoint_status"] in {"BEST SAVED", "LAST UPDATED"}
    output = capsys.readouterr().out
    assert "train_loss=" in output and "val_loss=" in output and "macro_f1=" in output and "early_stop=" in output
    loaded_json = load_training_history(manager.metrics_path)
    loaded_csv = load_training_history(manager.metrics_csv_path)
    assert len(loaded_json) == len(loaded_csv) == 1
    review = history_review(loaded_csv)
    assert len(review["all_epochs"]) == len(review["best_five_by_val_loss"]) == len(review["last_epoch"]) == 1


def test_persistence_deduplicates_epochs_for_resume_safety(tmp_path: Path) -> None:
    records = [
        {"epoch": 0, "train_loss": 1.2, "train_accuracy": 0.4, "val_loss": 1.1, "val_accuracy": 0.5, "val_macro_f1": 0.4},
        {"epoch": 0, "train_loss": 1.0, "train_accuracy": 0.6, "val_loss": 0.9, "val_accuracy": 0.7, "val_macro_f1": 0.6},
        {"epoch": 1, "train_loss": 0.9, "train_accuracy": 0.7, "val_loss": 0.8, "val_accuracy": 0.8, "val_macro_f1": 0.7},
    ]
    normalized = persist_training_history(
        records,
        json_path=tmp_path / "history.json",
        csv_path=tmp_path / "history.csv",
        jsonl_path=tmp_path / "history.jsonl",
    )
    assert [record["epoch"] for record in normalized] == [0, 1]
    assert normalized[0]["val_loss"] == 0.9
    assert len((tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()) == 2
