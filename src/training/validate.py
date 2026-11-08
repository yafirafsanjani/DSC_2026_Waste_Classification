"""Deterministic validation utilities; internal test remains caller-controlled."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.evaluation.metrics import compute_classification_metrics


def validate_one_epoch(model: nn.Module, data_loader: DataLoader, criterion: nn.Module,
                       class_mapping: Mapping[str, int], device: torch.device | str = "cpu", *,
                       show_progress: bool = False) -> dict[str, object]:
    """Evaluate every sample without gradients, optimizer changes, or checkpoint selection."""
    if len(data_loader.dataset) == 0:
        raise ValueError("Cannot validate an empty data loader.")
    target_device = torch.device(device)
    model.eval()
    total_loss, total_samples, all_targets, all_predictions = 0.0, 0, [], []
    iterator = tqdm(data_loader, desc="Validation", leave=False, disable=not show_progress)
    with torch.inference_mode():
        for inputs, targets in iterator:
            inputs, targets = inputs.to(target_device), targets.to(target_device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Validation loss is non-finite.")
            batch_size = targets.size(0)
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size
            all_targets.extend(targets.cpu().tolist())
            all_predictions.extend(outputs.argmax(dim=1).cpu().tolist())
    metrics = compute_classification_metrics(all_targets, all_predictions, class_mapping)
    metrics["loss"] = total_loss / total_samples
    return metrics
