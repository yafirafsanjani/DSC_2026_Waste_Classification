"""Explicit checkpoint restoration for a user-initiated resumed experiment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from src.callbacks.model_checkpoint import load_checkpoint, restore_rng_state


@dataclass(frozen=True)
class ResumeState:
    start_epoch: int
    best_epoch: int | None
    best_metric: float | None
    history: list[Mapping[str, Any]]
    training_config: Mapping[str, Any]


def _assert_compatible(payload: Mapping[str, Any], *, model_identifier: str | None,
                       class_mapping: Mapping[str, int] | None, training_config: Mapping[str, Any] | None) -> None:
    if model_identifier is not None and payload.get("model_identifier") != model_identifier:
        raise ValueError(f"Checkpoint model identifier mismatch: {payload.get('model_identifier')} != {model_identifier}")
    if class_mapping is not None and dict(payload.get("class_mapping", {})) != dict(class_mapping):
        raise ValueError("Checkpoint class mapping is incompatible with the active dataset.")
    if training_config is not None and dict(payload.get("training_config", {})) != dict(training_config):
        raise ValueError("Checkpoint training configuration is incompatible with this experiment.")


def resume_from_checkpoint(
    checkpoint_path: Path | str, model: nn.Module, optimizer: torch.optim.Optimizer, *,
    scheduler: object | None = None, scaler: torch.amp.GradScaler | None = None,
    early_stopping: Any | None = None, model_identifier: str | None = None,
    class_mapping: Mapping[str, int] | None = None, training_config: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu", restore_rng: bool = True,
) -> ResumeState:
    """Restore state only; callers decide whether and when to continue training."""
    payload = load_checkpoint(checkpoint_path, map_location)
    _assert_compatible(payload, model_identifier=model_identifier, class_mapping=class_mapping, training_config=training_config)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    if early_stopping is not None and payload.get("early_stopping_state"):
        early_stopping.load_state_dict(payload["early_stopping_state"])
    if restore_rng and payload.get("rng_state"):
        restore_rng_state(payload["rng_state"])
    return ResumeState(int(payload["epoch"]) + 1, payload.get("best_epoch"), payload.get("best_metric"),
                       list(payload.get("history", [])), dict(payload.get("training_config", {})))
