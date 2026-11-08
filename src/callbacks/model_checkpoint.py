"""Atomic best/last checkpoint persistence for reproducible notebook experiments."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    return path


def load_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")
    return payload


@dataclass
class ModelCheckpoint:
    directory: Path | str
    experiment_id: str
    monitor: str = "val_loss"
    mode: str = "min"
    best_metric: float | None = None
    best_epoch: int | None = None

    def __post_init__(self) -> None:
        self.directory = Path(self.directory) / self.experiment_id
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.mode not in {"min", "max"}:
            raise ValueError("Checkpoint mode must be 'min' or 'max'.")

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    @property
    def last_path(self) -> Path:
        return self.directory / "last.pt"

    def is_improved(self, value: float) -> bool:
        return self.best_metric is None or (value < self.best_metric if self.mode == "min" else value > self.best_metric)

    def payload(self, *, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                training_config: Mapping[str, Any], class_mapping: Mapping[str, int], model_identifier: str,
                scheduler: object | None = None, scaler: torch.amp.GradScaler | None = None,
                early_stopping_state: Mapping[str, Any] | None = None, history: list[Mapping[str, Any]] | None = None,
                seed: int = 42, metric: float | None = None) -> dict[str, Any]:
        return {
            "format_version": 1, "epoch": epoch, "metric": metric, "best_epoch": self.best_epoch,
            "best_metric": self.best_metric, "monitor": self.monitor, "mode": self.mode,
            "model_identifier": model_identifier, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "early_stopping_state": dict(early_stopping_state or {}), "training_config": dict(training_config),
            "class_mapping": dict(class_mapping), "history": list(history or []), "seed": seed,
            "rng_state": capture_rng_state(),
        }

    def save_last(self, payload: Mapping[str, Any]) -> Path:
        return atomic_torch_save(payload, self.last_path)

    def save_best_if_improved(self, value: float, epoch: int, payload: Mapping[str, Any]) -> Path | None:
        if not self.is_improved(value):
            return None
        self.best_metric, self.best_epoch = float(value), epoch
        saved = dict(payload)
        saved.update({"best_metric": self.best_metric, "best_epoch": self.best_epoch})
        return atomic_torch_save(saved, self.best_path)
