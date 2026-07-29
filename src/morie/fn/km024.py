# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.24: the shuffled token detection (STD) discriminative loss."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_std_loss"]


def _validate_probs(p, name):
    p = np.atleast_1d(np.asarray(p, dtype=float))
    if len(p) == 0:
        raise ValueError(f"{name} is empty.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError(f"every entry of {name} must lie in [0, 1].")
    return p


def kamath_ch2_std_loss(xhat, d):
    """L = -(1/|x_hat|) sum_i log P(d_i | x_hat_i).

    ``xhat`` holds the model's probability that each corrupted-input
    token is ORIGINAL; ``d`` the true labels (1 original, 0 replaced).
    The scored probability is p_i when d_i = 1 and 1 - p_i when
    d_i = 0 -- a model that always says "original" is heavily punished
    on the replaced tokens, which is the discriminator's whole job.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.24, printed
    p. 52.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch2_std_loss([0.5, 0.5], [1, 0])
    >>> abs(out["estimate"] - math.log(2)) < 1e-12
    True
    """
    p = _validate_probs(xhat, "xhat")
    d = np.atleast_1d(np.asarray(d)).astype(int)
    if len(d) != len(p):
        raise ValueError("need one label per token.")
    if np.any((d != 0) & (d != 1)):
        raise ValueError("labels must be 0 (replaced) or 1 (original).")
    scored = np.where(d == 1, p, 1.0 - p)
    with np.errstate(divide="ignore"):
        losses = -np.log(scored)
    return RichResult(payload={
        "estimate": float(np.mean(losses)),
        "per_token": [float(v) for v in losses],
        "accuracy": float(np.mean((p >= 0.5) == (d == 1))), "n": len(p),
        "method": "shuffled token detection (STD) loss (Kamath Eq 2.24)"})


def cheatsheet():
    return "km024: per-token binary CE against original/replaced labels"
