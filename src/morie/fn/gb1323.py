# morie.fn -- function file (rootcoder007/morie)
"""ARE invariance for two-sided tests."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_are_twosided"]


def gibbons_are_twosided(efficacy_T, efficacy_T_star):
    r"""Theorem 13.2.3: the ARE for two-sided rejection regions is the
    SAME squared-efficacy ratio as the one-sided case,

    .. math:: \mathrm{ARE}(T, T^*) =
              \left[\frac{c(T)}{c(T^*)}\right]^2,

    because both tails of an asymptotically normal statistic shift by
    the same local amount -- sidedness cancels from the ratio. The
    one- and two-sided values are returned together so the equality
    is explicit.

    Parameters
    ----------
    efficacy_T, efficacy_T_star : float > 0
        Efficacies of the two tests.

    Returns
    -------
    RichResult
        keys: ``are_two_sided``, ``are_one_sided``, ``equal``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 13.2.3.
    """
    cT = float(efficacy_T)
    cS = float(efficacy_T_star)
    if cT <= 0 or cS <= 0:
        raise ValueError("efficacies must be positive.")
    are = (cT / cS) ** 2
    return RichResult(
        payload={
            "are_two_sided": float(are), "are_one_sided": float(are),
            "equal": True,
            "method": "Two-sided ARE equals the one-sided ratio (Theorem 13.2.3)",
        }
    )


def cheatsheet():
    return "gb1323: sidedness cancels; same squared-efficacy ratio"
