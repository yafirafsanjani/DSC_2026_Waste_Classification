"""Serializable early-stopping state for user-controlled training sessions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass
class EarlyStopping:
    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 10
    min_delta: float = 0.001
    best_metric: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement: int = 0
    stopped: bool = False
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'.")
        if self.patience < 0 or self.min_delta < 0:
            raise ValueError("patience and min_delta must be non-negative.")

    def improved(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        return value < self.best_metric - self.min_delta if self.mode == "min" else value > self.best_metric + self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Record a completed epoch and return whether the caller should stop."""
        if self.stopped:
            return True
        if self.improved(value):
            self.best_metric, self.best_epoch, self.epochs_without_improvement = float(value), epoch, 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.stopped = True
                self.stop_reason = f"{self.monitor} did not improve by {self.min_delta} for {self.patience} epochs"
        return self.stopped

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = set(self.state_dict())
        missing = expected.difference(state)
        if missing:
            raise ValueError(f"EarlyStopping state is missing keys: {sorted(missing)}")
        for key in expected:
            setattr(self, key, state[key])
        self.__post_init__()
