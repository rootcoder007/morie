# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 5: the 3H (helpful, harmless, honest) alignment score."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_3h_alignment"]


def kamath_3h_alignment(helpful_score, harmless_score, honest_score,
                        weights=None):
    r"""score_3H = w_H*helpful + w_A*harmless + w_O*honest.

    The three rubric scores may be scalars or equal-length arrays of
    per-response ratings. ``weights`` defaults to the equal split
    (1/3, 1/3, 1/3); it is used AS GIVEN (not renormalized), because
    silently rescaling a caller's rubric would change the scale their
    thresholds are set on -- but the weight sum is reported so an
    unintended one is visible.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Alignment Tuning
    (3H); Askell et al. (2021).

    Examples
    --------
    >>> out = kamath_3h_alignment(0.8, 0.6, 1.0, [0.5, 0.3, 0.2])
    >>> round(out["estimate"], 12)     # 0.40 + 0.18 + 0.20
    0.78
    """
    h = np.atleast_1d(np.asarray(helpful_score, dtype=float))
    a = np.atleast_1d(np.asarray(harmless_score, dtype=float))
    o = np.atleast_1d(np.asarray(honest_score, dtype=float))
    if not (h.shape == a.shape == o.shape):
        raise ValueError(
            f"the three rubric scores must line up; got shapes "
            f"{h.shape}, {a.shape} and {o.shape}.")
    if h.size == 0:
        raise ValueError("no responses were scored.")
    w = np.array([1 / 3, 1 / 3, 1 / 3]) if weights is None else \
        np.atleast_1d(np.asarray(weights, dtype=float))
    if w.size != 3:
        raise ValueError(f"3H needs exactly 3 weights; got {w.size}.")
    if np.any(w < 0):
        raise ValueError("3H weights cannot be negative.")
    if w.sum() <= 0:
        raise ValueError("the 3H weights are all zero.")
    per = w[0] * h + w[1] * a + w[2] * o
    est = float(per[0]) if per.size == 1 else [float(v) for v in per]
    return RichResult(payload={
        "estimate": est, "score": [float(v) for v in per],
        "weights": [float(v) for v in w], "weight_sum": float(w.sum()),
        "n": int(per.size),
        "method": "3H alignment score (Kamath Ch 5)"})


def cheatsheet():
    return "km3h: weighted helpful/harmless/honest rubric score"
