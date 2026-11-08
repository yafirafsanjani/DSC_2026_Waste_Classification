"""Scheduler construction and correct epoch-end stepping semantics."""
from __future__ import annotations

from typing import Any, Mapping

import torch


def build_scheduler(optimizer: torch.optim.Optimizer, config: Mapping[str, Any] | None = None) -> tuple[object | None, str | None]:
    """Build a supported scheduler without performing a scheduler step."""
    options = dict(config or {})
    name = str(options.pop("name", "reduce_on_plateau")).lower()
    options.pop("monitor", None)
    if name in {"none", "null", ""}:
        return None, None
    if name in {"reduce_on_plateau", "reducelronplateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=options.pop("mode", "min"), factor=float(options.pop("factor", 0.5)),
            patience=int(options.pop("patience", 4)), min_lr=float(options.pop("min_lr", 1e-6)), **options,
        ), "reduce_on_plateau"
    if name in {"cosine", "cosineannealinglr"}:
        if "t_max" not in options:
            raise ValueError("CosineAnnealingLR requires t_max.")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(options.pop("t_max")), **options), "cosine"
    raise ValueError(f"Unsupported scheduler: {name}")


def step_scheduler(scheduler: object | None, scheduler_kind: str | None, monitor_value: float | None = None) -> None:
    """Step plateau schedulers with their monitored metric and others once per epoch."""
    if scheduler is None:
        return
    if scheduler_kind == "reduce_on_plateau":
        if monitor_value is None:
            raise ValueError("ReduceLROnPlateau requires a monitor value.")
        scheduler.step(monitor_value)
        return
    scheduler.step()


def current_learning_rates(optimizer: torch.optim.Optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]
