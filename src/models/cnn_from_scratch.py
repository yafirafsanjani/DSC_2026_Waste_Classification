"""CNN From Scratch architecture used by the baseline checkpoint."""
from __future__ import annotations

import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Create one of the baseline's two-convolution feature blocks."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class CNNFromScratch(nn.Module):
    """Four-block CNN matching `cnn_scratch_baseline_v1` exactly."""

    def __init__(self, num_classes: int, dropout: float = 0.40) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 192),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(192, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))
