from pathlib import Path

import pytest
import torch
from torch import nn

from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.lr_scheduler import build_scheduler, current_learning_rates, step_scheduler
from src.callbacks.model_checkpoint import ModelCheckpoint, load_checkpoint
from src.training.resume import resume_from_checkpoint


def _state(tmp_path: Path):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    checkpoints = ModelCheckpoint(tmp_path, "synthetic", best_metric=None)
    return model, optimizer, checkpoints


def test_early_stopping_tracks_patience_and_serializes() -> None:
    stopping = EarlyStopping(patience=2, min_delta=0.1)
    assert not stopping.step(1.0, 0)
    assert not stopping.step(0.95, 1)
    assert stopping.step(0.94, 2)
    clone = EarlyStopping()
    clone.load_state_dict(stopping.state_dict())
    assert clone.stopped and clone.best_epoch == 0


def test_checkpoint_best_last_and_resume_state(tmp_path: Path) -> None:
    model, optimizer, checkpoints = _state(tmp_path)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    payload = checkpoints.payload(model=model, optimizer=optimizer, epoch=3, training_config={"lr": 0.1},
                                  class_mapping={"a": 0, "b": 1}, model_identifier="tiny", seed=42)
    assert checkpoints.save_best_if_improved(0.5, 3, payload) == checkpoints.best_path
    checkpoints.save_last(payload)
    assert load_checkpoint(checkpoints.best_path)["best_metric"] == 0.5
    restored_model = nn.Linear(2, 2)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1, momentum=0.9)
    state = resume_from_checkpoint(checkpoints.last_path, restored_model, restored_optimizer,
                                   model_identifier="tiny", class_mapping={"a": 0, "b": 1}, training_config={"lr": 0.1})
    assert state.start_epoch == 4
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(expected, actual)
    with pytest.raises(ValueError, match="identifier mismatch"):
        resume_from_checkpoint(checkpoints.last_path, restored_model, restored_optimizer, model_identifier="other")


def test_scheduler_uses_correct_step_signature() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler, kind = build_scheduler(optimizer, {"name": "reduce_on_plateau", "monitor": "val_loss", "patience": 0, "factor": 0.5})
    step_scheduler(scheduler, kind, 1.0)
    step_scheduler(scheduler, kind, 1.0)
    assert current_learning_rates(optimizer)[0] == pytest.approx(0.05)
    cosine, kind = build_scheduler(optimizer, {"name": "cosine", "t_max": 2})
    optimizer.step()
    step_scheduler(cosine, kind)
