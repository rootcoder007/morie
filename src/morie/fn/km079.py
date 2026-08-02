# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.3: AlignScore's three-head joint loss."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_alignscore_total_loss"]


def kamath_ch6_alignscore_total_loss(L_3way, L_bin, L_reg, lambdas):
    """L_total = lam1 L_3way + lam2 L_bin + lam3 L_reg.

    One backbone, three heads, one weighted sum. ``lambdas`` must hold
    exactly three finite non-negative weights (a negative weight would
    reward a head for getting WORSE, which is never the intent) in the
    order (3way, bin, reg). Per-term contributions are returned so a
    head that dominates the total is visible.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.3, printed
    p. 222.

    Examples
    --------
    >>> out = kamath_ch6_alignscore_total_loss(1.0, 2.0, 3.0,
    ...                                        [0.5, 0.25, 0.25])
    >>> out["estimate"], out["contributions"]
    (1.75, [0.5, 0.5, 0.75])
    """
    lam = np.atleast_1d(np.asarray(lambdas, dtype=float))
    if lam.size != 3:
        raise ValueError(
            f"lambdas must hold exactly three weights; got {lam.size}.")
    if np.any(lam < 0) or not np.all(np.isfinite(lam)):
        raise ValueError("every weight must be finite and non-negative.")
    losses = np.asarray([float(L_3way), float(L_bin), float(L_reg)],
                        dtype=float)
    if not np.all(np.isfinite(losses)):
        raise ValueError("every component loss must be finite.")
    contrib = lam * losses
    return RichResult(payload={
        "estimate": float(contrib.sum()),
        "contributions": [float(v) for v in contrib],
        "losses": [float(v) for v in losses],
        "lambdas": [float(v) for v in lam], "n": 3,
        "method": "AlignScore joint loss (Kamath Eq 6.3)"})


def cheatsheet():
    return "km079: lam1 L_3way + lam2 L_bin + lam3 L_reg"
