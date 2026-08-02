# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 1: the emergent-ability step at a scale threshold."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_emergent_abilities"]


def kamath_emergent_abilities(scales, scores, threshold):
    r"""metric(N) = H(N - N_threshold) * f(N): a step, then the score.

    Below the threshold the Heaviside gate zeroes the metric; at or
    above it the measured score passes through. What makes a claim of
    emergence testable is the JUMP -- the mean score above the
    threshold minus the mean below -- which is the reported estimate.
    Both sides must be non-empty: a threshold with nothing on one side
    of it demonstrates nothing.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 1, Emergent
    Abilities; Wei et al. (2022).

    Examples
    --------
    >>> out = kamath_emergent_abilities([1.0, 10.0, 100.0],
    ...                                 [0.1, 0.1, 0.9], 50.0)
    >>> round(out["estimate"], 12)      # 0.9 - 0.1
    0.8
    >>> out["emergent_score"]
    [0.0, 0.0, 0.9]
    """
    N = np.atleast_1d(np.asarray(scales, dtype=float))
    f = np.atleast_1d(np.asarray(scores, dtype=float))
    if N.shape != f.shape:
        raise ValueError(
            f"{N.size} scales but {f.size} scores.")
    if N.size == 0:
        raise ValueError("no model scales were given.")
    thr = float(threshold)
    above = N >= thr
    if not above.any() or above.all():
        raise ValueError(
            f"the threshold {thr} puts every model on one side; an "
            "emergence jump needs models above AND below it.")
    gated = np.where(above, f, 0.0)
    jump = float(f[above].mean() - f[~above].mean())
    return RichResult(payload={
        "estimate": jump, "jump": jump,
        "emergent_score": [float(v) for v in gated],
        "mean_above": float(f[above].mean()),
        "mean_below": float(f[~above].mean()),
        "n_above": int(above.sum()), "threshold": thr, "n": int(N.size),
        "method": "emergent-ability step metric (Kamath Ch 1)"})


def cheatsheet():
    return "kmemer: Heaviside-gated scores plus the above/below jump"
