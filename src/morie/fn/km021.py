# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.21: the causal language modelling (CLM) loss."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_clm_loss"]


def _validate_probs(p, name):
    p = np.atleast_1d(np.asarray(p, dtype=float))
    if len(p) == 0:
        raise ValueError(f"{name} is empty.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError(f"every entry of {name} must lie in [0, 1].")
    return p


def kamath_ch2_clm_loss(x):
    """L = -(1/|x|) sum_i log P_i, every position scored.
    ``x`` holds P(x_i | x_<i) for every position under the caller's model.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.21, printed
    p. 51.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch2_clm_loss([0.5, 0.5])
    >>> abs(out["estimate"] - math.log(2)) < 1e-12
    True
    """
    p = _validate_probs(x, "x")
    with np.errstate(divide="ignore"):
        losses = -np.log(p)
    return RichResult(payload={
        "estimate": float(np.mean(losses)),
        "per_position": [float(v) for v in losses], "n": len(p),
        "method": "causal language modelling (CLM) loss (Kamath Eq 2.21)"})


def cheatsheet():
    return "km021: -mean log P over every position"
