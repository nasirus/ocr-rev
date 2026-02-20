"""
MicroOCR model definition (PyTorch — used for training only).

Architecture:
    4-layer CNN → collapse height → 2 linear layers → CTC output

    Input:  (B, 1, 32, W)   — batch of grayscale line images
    Output: (T, B, n_class)  — per-timestep class logits for CTC

Total parameters: ~101K  (~400KB float32, ~100KB int8)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Character set:
#   - letters + digits
#   - whitespace (space)
#   - common special characters
_ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_WHITESPACE = " "
_SPECIAL = ".,:;!?@#$%&*+-_/()[]{}'\"=<>"
CHARS = _ALNUM + _WHITESPACE + _SPECIAL
BLANK_IDX = len(CHARS)
NUM_CLASSES = len(CHARS) + 1


def char_to_index(c: str) -> int:
    """Map a character to its class index."""
    idx = CHARS.find(c)
    if idx == -1:
        raise ValueError(f"Character '{c}' not in alphabet")
    return idx


def index_to_char(idx: int) -> str:
    """Map a class index back to a character."""
    if idx == BLANK_IDX:
        return ""
    if 0 <= idx < len(CHARS):
        return CHARS[idx]
    raise ValueError(f"Index {idx} out of range")


class MicroOCRModel(nn.Module):
    """Tiny CNN + CTC recognition model.

    The network processes a (B, 1, 32, W) image and outputs
    (T, B, num_classes) logits where T = W // 4 (due to two 2x2 max-pools).
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor: 4 conv layers, 2 max-pools
        # Input: (B, 1, 32, W)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # → (B,16,32,W)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # → (B,32,16,W/2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # → (B,64,8,W/4)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # → (B,64,8,W/4)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)

        # After conv4: (B, 64, 8, W/4)
        # Collapse height: reshape to (B, W/4, 64*8) = (B, T, 512)
        self.fc1 = nn.Linear(64 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 1, 32, W) float tensor, values in [0, 1].

        Returns:
            (T, B, num_classes) log-probabilities for CTC.
        """
        # Conv block 1: conv → BN → relu → pool
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # (B,16,16,W/2)
        # Conv block 2: conv → BN → relu → pool
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # (B,32,8,W/4)
        # Conv block 3: conv → BN → relu (no pool)
        x3 = self.relu(self.bn3(self.conv3(x)))  # (B,64,8,W/4)
        # Conv block 4: conv → BN → relu + residual from conv3
        x = self.relu(self.bn4(self.conv4(x3)) + x3)  # (B,64,8,W/4)

        # Collapse spatial dims: (B, 64, 8, T) → (B, T, 64*8)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, T, 64, 8)
        x = x.reshape(b, w, c * h)  # (B, T, 512)

        # Classifier
        x = self.dropout(self.relu(self.fc1(x)))  # (B, T, 128)
        x = self.fc2(x)  # (B, T, num_classes)

        # CTC expects (T, B, C)
        x = x.permute(1, 0, 2)  # (T, B, num_classes)
        return x

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
