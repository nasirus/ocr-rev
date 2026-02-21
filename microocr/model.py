"""
MicroOCR model definition (PyTorch — used for training only).

Architecture:
    4-layer CNN → collapse height → BiGRU → 2 linear layers → CTC output

    Input:  (B, 1, H, W)    — batch of grayscale line images
    Output: (T, B, n_class)  — per-timestep class logits for CTC

Total parameters: ~1.28M  (~5MB float32)
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
    """Tiny CNN + BiGRU + CTC recognition model.

    The network processes a (B, 1, H, W) image and outputs
    (T, B, num_classes) logits where T = W // 4 (due to two 2x2 max-pools).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 40,
        head_hidden: int = 384,
        rnn_hidden: int = 96,
    ):
        super().__init__()
        self.num_classes = num_classes

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # Feature extractor: 4 conv layers, 2 max-pools
        # Input: (B, 1, H, W)
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)  # → (B,c1,H,W)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)  # → (B,c2,H/2,W/2)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)  # → (B,c3,H/4,W/4)
        self.bn3 = nn.BatchNorm2d(c3)
        self.conv4 = nn.Conv2d(c3, c3, kernel_size=3, padding=1)  # → (B,c3,H/4,W/4)
        self.bn4 = nn.BatchNorm2d(c3)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)

        # After conv4: (B, c3, H/4, W/4)
        # Collapse height: reshape to (B, W/4, c3*H/4) = (B, T, collapse_dim)
        # For H=32: collapse_dim = 160*8 = 1280
        # The actual collapse_dim is computed dynamically in forward()

        # BiGRU for sequential context
        self.rnn_hidden = rnn_hidden
        self._collapse_dim = c3 * 8  # default for H=32, used for param init
        self.rnn = nn.GRU(
            input_size=self._collapse_dim,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # FC head takes BiGRU output (2 * rnn_hidden)
        self.fc1 = nn.Linear(rnn_hidden * 2, head_hidden)
        self.fc2 = nn.Linear(head_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 1, 32, W) float tensor, values in [0, 1].

        Returns:
            (T, B, num_classes) log-probabilities for CTC.
        """
        # Conv block 1: conv → BN → relu → pool
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # downsample x2
        # Conv block 2: conv → BN → relu → pool
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # downsample x2 again
        # Conv block 3: conv → BN → relu (no pool)
        x3 = self.relu(self.bn3(self.conv3(x)))  # (B,c3,8,W/4)
        # Conv block 4: conv → BN → relu + residual from conv3
        x = self.relu(self.bn4(self.conv4(x3)) + x3)  # (B,c3,8,W/4)

        # Collapse spatial dims: (B, c3, H', T) → (B, T, c3*H')
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, T, C, H')
        x = x.reshape(b, w, c * h)  # (B, T, C*H')

        # BiGRU for sequential context
        x, _ = self.rnn(x)  # (B, T, 2*rnn_hidden)

        # Classifier
        x = self.dropout(self.relu(self.fc1(x)))  # (B, T, hidden)
        x = self.fc2(x)  # (B, T, num_classes)

        # CTC expects (T, B, C)
        x = x.permute(1, 0, 2)  # (T, B, num_classes)
        return x

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
