"""
CTC (Connectionist Temporal Classification) decoder — pure NumPy.

Supports:
    - Greedy decoding (fast, good enough for most cases)
    - Beam search decoding (better accuracy, optional)
    - Top-K beam pruning for speed
    - Character bigram language model for disambiguation
"""

from __future__ import annotations

import numpy as np

from microocr.model import BLANK_IDX, CHARS

# Top-K characters to consider per timestep in beam search
_BEAM_TOP_K = 15

# Default LM weight for bigram scoring
_DEFAULT_LM_WEIGHT = 0.15


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
    bigram_log_probs: np.ndarray | None = None,
    lm_weight: float = _DEFAULT_LM_WEIGHT,
) -> str:
    """Beam search CTC decoding with top-K pruning and optional bigram LM.

    More accurate than greedy but slower. Uses prefix beam search
    with top-K character pruning per timestep for speed.

    Args:
        logits: 2-D array of shape (T, num_classes).
        beam_width: Number of beams to maintain.
        bigram_log_probs: Optional NxN bigram log-probability table
            where N == len(CHARS).
            If provided, beam scores are biased by character bigram probs.
        lm_weight: Weight for bigram LM scoring (only used if bigram_log_probs
            is provided).

    Returns:
        Decoded string (best beam).
    """
    T, C = logits.shape
    if C != len(CHARS) + 1:
        raise ValueError(
            f"Expected logits with {len(CHARS) + 1} classes, got shape {logits.shape}"
        )
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")

    # Softmax over classes
    log_probs = _log_softmax(logits)

    # Pre-compute top-K character indices per timestep (excluding blank)
    # We always include blank, plus top-K non-blank characters
    top_k = min(_BEAM_TOP_K, len(CHARS))
    top_k_indices = np.argpartition(log_probs[:, :BLANK_IDX], -top_k, axis=1)[
        :, -top_k:
    ]  # (T, top_k)

    # Each beam: (prefix_tuple, (log_prob_blank, log_prob_nonblank))
    # Start with empty prefix
    beams: dict[tuple[int, ...], tuple[float, float]] = {
        (): (0.0, float("-inf")),  # (p_blank, p_nonblank)
    }

    for t in range(T):
        new_beams: dict[tuple[int, ...], tuple[float, float]] = {}
        active_chars = top_k_indices[t]  # top-K non-blank chars at this timestep

        for prefix, (p_b, p_nb) in beams.items():
            p_total = _log_add(p_b, p_nb)

            # Extend with blank
            _add_beam(
                new_beams,
                prefix,
                p_total + log_probs[t, BLANK_IDX],
                float("-inf"),
            )

            # Extend with top-K characters only
            for c in active_chars:
                c = int(c)
                lp = log_probs[t, c]

                # Apply bigram LM bias if available
                if bigram_log_probs is not None and prefix:
                    prev_char = prefix[-1]
                    lp = lp + lm_weight * bigram_log_probs[prev_char, c]

                if prefix and prefix[-1] == c:
                    # CTC repeated-char rule:
                    # - keep same prefix only from nonblank path
                    _add_beam(new_beams, prefix, float("-inf"), p_nb + lp)
                    # - append repeated char only from blank path
                    new_prefix = prefix + (c,)
                    _add_beam(new_beams, new_prefix, float("-inf"), p_b + lp)
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


def build_bigram_table(texts: list[str]) -> np.ndarray:
    """Build a character bigram log-probability table from text samples.

    Args:
        texts: List of text strings to compute bigram statistics from.

    Returns:
        NxN float32 array of log-probabilities where N == len(CHARS).
    """
    n = len(CHARS)
    counts = np.ones((n, n), dtype=np.float32)  # Laplace smoothing

    for text in texts:
        for i in range(len(text) - 1):
            a = CHARS.find(text[i])
            b = CHARS.find(text[i + 1])
            if a >= 0 and b >= 0:
                counts[a, b] += 1.0

    # Normalize rows to probabilities, then take log
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = counts / row_sums
    return np.log(probs).astype(np.float32)


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
