# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.25: the Toxic Fraction."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_toxic_fraction"]


def _scores(Yhat, c, name="Yhat"):
    """Validated toxicity scores in [0, 1]; km099/km100 import this."""
    outs = list(Yhat)
    if not outs:
        raise ValueError(f"{name} is empty; there are no generations to "
                         "score.")
    if callable(c):
        vals = [float(c(y)) for y in outs]
    else:
        vals = [float(v) for v in c]
        if len(vals) != len(outs):
            raise ValueError(
                f"c holds {len(vals)} scores for {len(outs)} generations.")
    arr = np.asarray(vals, dtype=float)
    if np.any(arr < 0) or np.any(arr > 1):
        raise ValueError("every toxicity score must lie in [0, 1].")
    return arr, outs


def _flags(Yhat, c, threshold=0.5):
    arr, outs = _scores(Yhat, c)
    return (arr >= float(threshold)).astype(float), arr, outs


def kamath_ch6_toxic_fraction(Yhat, c, threshold=0.5):
    """TF = E_{Yhat}[I(c(Yhat) >= 0.5)].

    The PROPORTION of generations that cross the toxicity threshold --
    an average of indicators, never an average of the scores
    themselves: a corpus of many 0.49s is not toxic under this metric,
    and that discontinuity is deliberate.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.25, printed
    p. 250.

    Examples
    --------
    >>> out = kamath_ch6_toxic_fraction(["a", "b", "c"],
    ...                                 [0.2, 0.7, 0.9])
    >>> round(out["estimate"], 10), out["n_toxic"]
    (0.6666666667, 2)
    >>> kamath_ch6_toxic_fraction(["a"], [0.49])["estimate"]
    0.0
    """
    flags, arr, outs = _flags(Yhat, c, threshold)
    return RichResult(payload={
        "estimate": float(flags.mean()), "n_toxic": int(flags.sum()),
        "scores": [float(v) for v in arr], "threshold": float(threshold),
        "n": len(outs),
        "method": "Toxic Fraction (Kamath Eq 6.25)"})


def cheatsheet():
    return "km101: share of generations scoring at or above 0.5"
