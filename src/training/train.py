"""One-epoch custom PyTorch training loop; it never starts training on import."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def _make_scaler(enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=enabled)


def train_one_epoch(model: nn.Module, data_loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device | str = "cpu", *,
                    scaler: torch.amp.GradScaler | None = None, amp_enabled: bool = True,
                    gradient_accumulation_steps: int = 1, gradient_clip_norm: float | None = None,
                    show_progress: bool = True, epoch: int | None = None,
                    max_epochs: int | None = None) -> tuple[dict[str, float | int], torch.amp.GradScaler | None]:
    """Train one epoch with sample-weighted metrics and correct final accumulation window."""
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if len(data_loader) == 0:
        raise ValueError("Cannot train on an empty data loader.")
    target_device = torch.device(device)
    use_amp = amp_enabled and target_device.type == "cuda" and torch.cuda.is_available()
    active_scaler = scaler or _make_scaler(use_amp)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    batch_count = len(data_loader)
    description = "Training" if epoch is None or max_epochs is None else f"Epoch {epoch + 1}/{max_epochs} train"
    iterator = tqdm(data_loader, total=batch_count, desc=description, leave=False, disable=not show_progress)
    smoothed_loss: float | None = None
    for batch_index, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(target_device), targets.to(target_device)
        window_size = min(gradient_accumulation_steps, batch_count - batch_index + (batch_index % gradient_accumulation_steps))
        with torch.autocast(device_type=target_device.type, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"Training loss is non-finite at batch {batch_index}.")
        batch_size = targets.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((outputs.detach().argmax(dim=1) == targets).sum())
        total_samples += batch_size
        batch_loss = float(loss.detach())
        smoothed_loss = batch_loss if smoothed_loss is None else 0.9 * smoothed_loss + 0.1 * batch_loss
        if show_progress:
            iterator.set_postfix(
                loss=f"{smoothed_loss:.4f}",
                acc=f"{total_correct / total_samples:.2%}",
                lr=f"{float(optimizer.param_groups[0]['lr']):.2e}",
            )
        active_scaler.scale(loss / window_size).backward()
        is_window_end = (batch_index + 1) % gradient_accumulation_steps == 0 or batch_index + 1 == batch_count
        if is_window_end:
            if gradient_clip_norm is not None:
                active_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            active_scaler.step(optimizer)
            active_scaler.update()
            optimizer.zero_grad(set_to_none=True)
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples,
            "sample_count": total_samples}, active_scaler
