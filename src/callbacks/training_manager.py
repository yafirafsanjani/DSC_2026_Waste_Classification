"""The high-level training coordinator; loops remain in training modules."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.lr_scheduler import current_learning_rates, step_scheduler
from src.callbacks.model_checkpoint import ModelCheckpoint
from src.evaluation.training_monitoring import format_epoch_summary, persist_training_history
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
    train_mode_callback: Callable[[nn.Module], None] | None = None

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
        self.metrics_csv_path = self.project_root / "outputs" / "metrics" / f"{self.experiment_id}.csv"
        self.log_path = self.project_root / "outputs" / "logs" / f"{self.experiment_id}.jsonl"

    def _persist_history(self) -> None:
        self.history = persist_training_history(
            self.history, json_path=self.metrics_path, csv_path=self.metrics_csv_path, jsonl_path=self.log_path,
        )

    def run_epoch(self, train_loader: DataLoader, validation_loader: DataLoader, epoch: int, *,
                  amp_enabled: bool = True, gradient_accumulation_steps: int = 1,
                  gradient_clip_norm: float | None = None, show_progress: bool = True,
                  max_epochs: int | None = None) -> dict[str, Any]:
        """Run exactly one train/validation epoch and persist actual resulting metrics."""
        started = time.perf_counter()
        train_metrics, self.scaler = train_one_epoch(
            self.model, train_loader, self.criterion, self.optimizer, self.device, scaler=self.scaler,
            amp_enabled=amp_enabled, gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clip_norm=gradient_clip_norm, show_progress=show_progress, epoch=epoch, max_epochs=max_epochs,
            train_mode_callback=self.train_mode_callback,
        )
        validation_metrics = validate_one_epoch(self.model, validation_loader, self.criterion, self.class_mapping, self.device)
        metric_name = self.early_stopping.monitor.removeprefix("val_")
        monitor_value = validation_metrics["loss"] if self.early_stopping.monitor == "val_loss" else validation_metrics.get(metric_name)
        if monitor_value is None:
            raise ValueError(f"Unsupported monitored metric: {self.early_stopping.monitor}")
        previous_learning_rates = current_learning_rates(self.optimizer)
        step_scheduler(self.scheduler, self.scheduler_kind, float(monitor_value))
        learning_rates = current_learning_rates(self.optimizer)
        stopped = self.early_stopping.step(float(monitor_value), epoch)
        record = {
            "experiment_id": self.experiment_id, "epoch": epoch,
            "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
            "val_loss": validation_metrics["loss"], "val_accuracy": validation_metrics["accuracy"],
            "val_macro_precision": validation_metrics["macro_precision"], "val_macro_recall": validation_metrics["macro_recall"],
            "val_macro_f1": validation_metrics["macro_f1"], "learning_rates": learning_rates,
            "learning_rate": learning_rates[0], "previous_learning_rates": previous_learning_rates,
            "learning_rate_changed": learning_rates != previous_learning_rates,
            "epoch_seconds": time.perf_counter() - started, "best_epoch": self.early_stopping.best_epoch,
            "best_validation_loss": self.early_stopping.best_metric,
            "early_stopping_counter": self.early_stopping.epochs_without_improvement,
            "early_stopping": stopped, "early_stopping_reason": self.early_stopping.stop_reason,
        }
        if any(int(existing["epoch"]) == epoch for existing in self.history):
            raise ValueError(f"Epoch {epoch} is already present in history; resume must start after the last persisted epoch.")
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
        record["checkpoint_status"] = "BEST SAVED" if best_path is not None else "LAST UPDATED"
        self.checkpoints.save_last(self.checkpoints.payload(
            model=self.model, optimizer=self.optimizer, epoch=epoch, training_config=self.training_config,
            class_mapping=self.class_mapping, model_identifier=self.model_identifier, scheduler=self.scheduler,
            scaler=self.scaler, early_stopping_state=self.early_stopping.state_dict(), history=self.history,
            seed=self.seed, metric=float(monitor_value),
        ))
        self._persist_history()
        print(format_epoch_summary(record, max_epochs=max_epochs))
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
                gradient_clip_norm=gradient_clip_norm, show_progress=show_progress, max_epochs=max_epochs,
            )
            if record["early_stopping"]:
                break
        return self.history
