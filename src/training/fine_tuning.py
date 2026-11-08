"""Architecture-agnostic fine-tuning preparation; no phase is run automatically."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from src.callbacks.model_checkpoint import load_checkpoint


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def freeze_backbone(backbone: nn.Module) -> None:
    set_module_trainable(backbone, False)


def unfreeze_backbone(backbone: nn.Module, *, last_n_children: int | None = None) -> None:
    """Unfreeze all children or only the final architecture-specific child modules."""
    if last_n_children is None:
        set_module_trainable(backbone, True)
        return
    if last_n_children <= 0:
        raise ValueError("last_n_children must be positive when provided.")
    children = list(backbone.children())
    if not children:
        set_module_trainable(backbone, True)
        return
    for child in children[-last_n_children:]:
        set_module_trainable(child, True)


def build_parameter_groups(backbone: nn.Module, classifier: nn.Module, *, backbone_lr: float,
                           classifier_lr: float, weight_decay: float = 0.0) -> list[dict[str, object]]:
    """Build non-overlapping parameter groups while respecting frozen parameters."""
    backbone_params = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    classifier_params = [parameter for parameter in classifier.parameters() if parameter.requires_grad]
    overlap = {id(parameter) for parameter in backbone_params}.intersection(id(parameter) for parameter in classifier_params)
    if overlap:
        raise ValueError("backbone and classifier parameter groups overlap.")
    groups: list[dict[str, object]] = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone"})
    if classifier_params:
        groups.append({"params": classifier_params, "lr": classifier_lr, "weight_decay": weight_decay, "name": "classifier"})
    if not groups:
        raise ValueError("No trainable parameters are available for fine-tuning.")
    return groups


@dataclass(frozen=True)
class FineTuningPlan:
    parent_experiment_id: str
    phase_name: str
    reset_optimizer: bool = True
    reset_scheduler: bool = True
    reset_early_stopping: bool = True


def create_fine_tuning_optimizer(optimizer_type: type[torch.optim.Optimizer], parameter_groups: Iterable[dict[str, object]], **kwargs: object) -> torch.optim.Optimizer:
    """Create fresh optimizer state for an explicitly initiated fine-tuning phase."""
    return optimizer_type(list(parameter_groups), **kwargs)


def load_best_checkpoint_for_fine_tuning(model: nn.Module, checkpoint_path: str, *,
                                         model_identifier: str, class_mapping: dict[str, int]) -> dict[str, object]:
    """Load user-selected best weights while leaving optimizer/callback state to a fresh phase."""
    payload = load_checkpoint(checkpoint_path)
    if payload.get("model_identifier") != model_identifier:
        raise ValueError("Best checkpoint model identifier is incompatible with the requested fine-tuning phase.")
    if dict(payload.get("class_mapping", {})) != dict(class_mapping):
        raise ValueError("Best checkpoint class mapping is incompatible with the active dataset.")
    model.load_state_dict(payload["model_state_dict"])
    return {"parent_epoch": payload.get("epoch"), "parent_best_epoch": payload.get("best_epoch"), "parent_best_metric": payload.get("best_metric")}
