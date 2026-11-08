"""The high-level training coordinator; loops remain in training modules."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.lr_scheduler import current_learning_rates, step_scheduler
from src.callbacks.model_checkpoint import ModelCheckpoint
from src.training.train import train_one_epoch
from src.training.validate import validate_one_epoch


@dataclass
class TrainingManager:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    class_mapping: Mapping[str, int]
    model_identifier: str
    experiment_id: str
    project_root: Path | str = Path.cwd()
    device: torch.device | str = "cpu"
    scheduler: object | None = None
    scheduler_kind: str | None = None
    early_stopping: EarlyStopping = field(default_factory=EarlyStopping)
    scaler: torch.amp.GradScaler | None = None
    training_config: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 42
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.device = torch.device(self.device)
        self.model.to(self.device)
        self.checkpoints = ModelCheckpoint(
            self.project_root / "models" / "checkpoints", self.experiment_id,
            monitor=self.early_stopping.monitor, mode=self.early_stopping.mode,
            best_metric=self.early_stopping.best_metric, best_epoch=self.early_stopping.best_epoch,
        )
        self.metrics_path = self.project_root / "outputs" / "metrics" / f"{self.experiment_id}.json"
        self.log_path = self.project_root / "outputs" / "logs" / f"{self.experiment_id}.jsonl"

    def _persist_history(self) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.history[-1]) + "\n")

    def run_epoch(self, train_loader: DataLoader, validation_loader: DataLoader, epoch: int, *,
                  amp_enabled: bool = True, gradient_accumulation_steps: int = 1,
                  gradient_clip_norm: float | None = None, show_progress: bool = True) -> dict[str, Any]:
        """Run exactly one train/validation epoch and persist actual resulting metrics."""
        started = time.perf_counter()
        train_metrics, self.scaler = train_one_epoch(
            self.model, train_loader, self.criterion, self.optimizer, self.device, scaler=self.scaler,
            amp_enabled=amp_enabled, gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clip_norm=gradient_clip_norm, show_progress=show_progress,
        )
        validation_metrics = validate_one_epoch(self.model, validation_loader, self.criterion, self.class_mapping, self.device)
        metric_name = self.early_stopping.monitor.removeprefix("val_")
        monitor_value = validation_metrics["loss"] if self.early_stopping.monitor == "val_loss" else validation_metrics.get(metric_name)
        if monitor_value is None:
            raise ValueError(f"Unsupported monitored metric: {self.early_stopping.monitor}")
        step_scheduler(self.scheduler, self.scheduler_kind, float(monitor_value))
        stopped = self.early_stopping.step(float(monitor_value), epoch)
        record = {
            "epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
            "val_loss": validation_metrics["loss"], "val_accuracy": validation_metrics["accuracy"],
            "val_macro_precision": validation_metrics["macro_precision"], "val_macro_recall": validation_metrics["macro_recall"],
            "val_macro_f1": validation_metrics["macro_f1"], "learning_rates": current_learning_rates(self.optimizer),
            "epoch_seconds": time.perf_counter() - started, "best_epoch": self.early_stopping.best_epoch,
            "early_stopping": stopped, "early_stopping_reason": self.early_stopping.stop_reason,
        }
        self.history.append(record)
        payload = self.checkpoints.payload(
            model=self.model, optimizer=self.optimizer, epoch=epoch, training_config=self.training_config,
            class_mapping=self.class_mapping, model_identifier=self.model_identifier, scheduler=self.scheduler,
            scaler=self.scaler, early_stopping_state=self.early_stopping.state_dict(), history=self.history,
            seed=self.seed, metric=float(monitor_value),
        )
        best_path = self.checkpoints.save_best_if_improved(float(monitor_value), epoch, payload)
        record["checkpoint_best_saved"] = best_path is not None
        record["checkpoint_last"] = str(self.checkpoints.last_path)
        self.checkpoints.save_last(self.checkpoints.payload(
            model=self.model, optimizer=self.optimizer, epoch=epoch, training_config=self.training_config,
            class_mapping=self.class_mapping, model_identifier=self.model_identifier, scheduler=self.scheduler,
            scaler=self.scaler, early_stopping_state=self.early_stopping.state_dict(), history=self.history,
            seed=self.seed, metric=float(monitor_value),
        ))
        self._persist_history()
        return record

    def fit(self, train_loader: DataLoader, validation_loader: DataLoader, max_epochs: int, *, start_epoch: int = 0,
            amp_enabled: bool = True, gradient_accumulation_steps: int = 1,
            gradient_clip_norm: float | None = None, show_progress: bool = True) -> list[dict[str, Any]]:
        """Explicitly run user-requested epochs; never resumes after an early stop automatically."""
        if max_epochs <= start_epoch:
            raise ValueError("max_epochs must be greater than start_epoch.")
        for epoch in range(start_epoch, max_epochs):
            record = self.run_epoch(
                train_loader, validation_loader, epoch, amp_enabled=amp_enabled,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_clip_norm=gradient_clip_norm, show_progress=show_progress,
            )
            if record["early_stopping"]:
                break
        return self.history
