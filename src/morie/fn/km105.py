# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.29: GeDi's combined training loss."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_gedi_combined_loss"]


def kamath_ch6_gedi_combined_loss(L_g, L_d, lam):
    """L_gd = lam L_g + (1 - lam) L_d.

    A CONVEX combination, so lam must lie in [0, 1]: the generative
    loss calibrates token probabilities, the discriminative loss
    sharpens class separation, and lam trades one against the other.
    lam = 1 trains a plain LM, lam = 0 a pure discriminator.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.29, printed
    p. 254.

    Examples
    --------
    >>> out = kamath_ch6_gedi_combined_loss(1.0, 3.0, 0.25)
    >>> out["estimate"], out["contributions"]
    (2.5, [0.25, 2.25])
    >>> kamath_ch6_gedi_combined_loss(1.0, 3.0, 1.0)["estimate"]
    1.0
    """
    lam = float(lam)
    if not (0.0 <= lam <= 1.0):
        raise ValueError(
            f"lam = {lam:.6g} is outside [0, 1]; Eq 6.29 is a convex "
            "combination.")
    lg, ld = float(L_g), float(L_d)
    if not (np.isfinite(lg) and np.isfinite(ld)):
        raise ValueError("both component losses must be finite.")
    contrib = [lam * lg, (1.0 - lam) * ld]
    return RichResult(payload={
        "estimate": float(sum(contrib)), "contributions": contrib,
        "L_g": lg, "L_d": ld, "lam": lam, "n": 2,
        "method": "GeDi combined loss (Kamath Eq 6.29)"})


def cheatsheet():
    return "km105: lam L_g + (1-lam) L_d, convex in lam"
