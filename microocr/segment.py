"""
Text line segmentation using horizontal projection profiles.

Takes a binary image and returns a list of cropped line images,
one per detected text line. No neural network needed — pure
histogram analysis.
"""

from __future__ import annotations

import numpy as np


def segment_lines(
    binary: np.ndarray,
    min_line_height: int = 8,
    gap_threshold: float = 0.99,
) -> list[np.ndarray]:
    """Segment a binary image into individual text lines.

    Uses a horizontal projection profile: sum each row's ink pixels,
    then find continuous runs of rows that contain ink (separated by
    blank gaps).

    Args:
        binary: 2-D uint8 binary image (0=ink, 255=background).
        min_line_height: Minimum height in pixels for a valid line.
        gap_threshold: Fraction of white pixels in a row to consider
            it a gap (0.99 = 99% white → gap row).

    Returns:
        List of 2-D uint8 arrays, one per text line (top to bottom).
        Returns the full image as a single-element list if no gaps found.
    """
    if binary.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {binary.shape}")

    h, w = binary.shape
    if h == 0 or w == 0:
        return []

    # Horizontal projection: fraction of white pixels per row
    white_fraction = np.sum(binary > 128, axis=1) / w

    # A row is a "gap" if it's mostly white
    is_gap = white_fraction >= gap_threshold

    # Find contiguous runs of non-gap (ink) rows
    lines: list[np.ndarray] = []
    in_line = False
    line_start = 0

    for y in range(h):
        if not is_gap[y] and not in_line:
            # Start of a new line
            in_line = True
            line_start = y
        elif is_gap[y] and in_line:
            # End of a line
            in_line = False
            if y - line_start >= min_line_height:
                lines.append(binary[line_start:y, :])

    # Handle line that extends to the bottom
    if in_line and h - line_start >= min_line_height:
        lines.append(binary[line_start:h, :])

    # If no lines detected, return the whole image as one line
    if not lines:
        lines = [binary]

    return lines
