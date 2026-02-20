"""
CTC (Connectionist Temporal Classification) decoder — pure NumPy.

Supports:
    - Greedy decoding (fast, good enough for most cases)
    - Beam search decoding (better accuracy, optional)
"""

from __future__ import annotations

import numpy as np


# Character set must match model.py
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
BLANK_IDX = len(CHARS)  # 62


def greedy_decode(logits: np.ndarray) -> str:
    """Greedy CTC decoding.

    Takes per-timestep logits and returns the decoded string by:
    1. Taking argmax at each timestep
    2. Collapsing consecutive duplicates
    3. Removing blank tokens

    Args:
        logits: 2-D array of shape (T, num_classes) — raw logits
            or log-probabilities.

    Returns:
        Decoded string.
    """
    # Argmax at each timestep
    best = np.argmax(logits, axis=1)  # (T,)

    # Collapse consecutive duplicates
    collapsed: list[int] = []
    prev = -1
    for idx in best:
        if idx != prev:
            collapsed.append(int(idx))
        prev = idx

    # Remove blanks and map to characters
    chars = []
    for idx in collapsed:
        if idx != BLANK_IDX and 0 <= idx < len(CHARS):
            chars.append(CHARS[idx])

    return "".join(chars)


def beam_decode(
    logits: np.ndarray,
    beam_width: int = 10,
) -> str:
    """Beam search CTC decoding.

    More accurate than greedy but slower. Uses prefix beam search.

    Args:
        logits: 2-D array of shape (T, num_classes).
        beam_width: Number of beams to maintain.

    Returns:
        Decoded string (best beam).
    """
    T, C = logits.shape

    # Softmax over classes
    log_probs = _log_softmax(logits)

    # Each beam: (prefix_tuple, (log_prob_blank, log_prob_nonblank))
    # Start with empty prefix
    beams: dict[tuple[int, ...], tuple[float, float]] = {
        (): (0.0, float("-inf")),  # (p_blank, p_nonblank)
    }

    for t in range(T):
        new_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        for prefix, (p_b, p_nb) in beams.items():
            p_total = _log_add(p_b, p_nb)

            # Extend with blank
            _add_beam(
                new_beams,
                prefix,
                p_total + log_probs[t, BLANK_IDX],
                float("-inf"),
            )

            # Extend with each character
            for c in range(len(CHARS)):
                lp = log_probs[t, c]

                if prefix and prefix[-1] == c:
                    # Same char as last: only extend from blank path
                    _add_beam(new_beams, prefix, float("-inf"), p_b + lp)
                    # Also allow new repeated char
                    new_prefix = prefix + (c,)
                    _add_beam(new_beams, new_prefix, float("-inf"), p_total + lp)
                else:
                    new_prefix = prefix + (c,)
                    _add_beam(new_beams, new_prefix, float("-inf"), p_total + lp)

        # Prune to beam_width
        scored = {k: _log_add(v[0], v[1]) for k, v in new_beams.items()}
        top_keys = sorted(scored, key=lambda k: scored[k], reverse=True)[:beam_width]
        beams = {k: new_beams[k] for k in top_keys}

    # Best beam
    best_prefix = max(beams, key=lambda k: _log_add(beams[k][0], beams[k][1]))
    return "".join(CHARS[c] for c in best_prefix)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along axis 1."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return shifted - log_sum_exp


def _log_add(a: float, b: float) -> float:
    """Stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


def _add_beam(
    beams: dict[tuple[int, ...], tuple[float, float]],
    prefix: tuple[int, ...],
    p_blank: float,
    p_nonblank: float,
) -> None:
    """Add or merge a beam entry."""
    if prefix in beams:
        old_b, old_nb = beams[prefix]
        beams[prefix] = (_log_add(old_b, p_blank), _log_add(old_nb, p_nonblank))
    else:
        beams[prefix] = (p_blank, p_nonblank)
