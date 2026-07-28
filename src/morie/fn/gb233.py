# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic normality of the EDF at a point."""

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["gibbons_edf_asymp_normal"]


def gibbons_edf_asymp_normal(S_n_x, F_x, n):
    r"""Theorem 2.3.3: at any fixed x with 0 < F(x) < 1,

    .. math:: Z = \frac{\sqrt{n}\,[S_n(x) - F(x)]}
              {\sqrt{F(x)(1 - F(x))}} \;\to_d\; N(0, 1),

    the de Moivre-Laplace CLT applied to the binomial count
    :math:`n S_n(x)`.

    Parameters
    ----------
    S_n_x : float in [0, 1]
        Observed EDF value at x.
    F_x : float in (0, 1)
        True CDF value.
    n : int
        Sample size.

    Returns
    -------
    RichResult
        keys: ``z``, ``p_two_sided``, ``se``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 2.3.3.
    """
    S = float(S_n_x)
    F = float(F_x)
    if not 0 <= S <= 1:
        raise ValueError(f"S_n(x) must lie in [0, 1], got {S}.")
    if not 0 < F < 1:
        raise ValueError(f"F(x) must lie strictly in (0, 1), got {F}.")
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    se = np.sqrt(F * (1 - F) / n)
    z = (S - F) / se
    return RichResult(
        payload={
            "z": float(z), "p_two_sided": float(2 * stats.norm.sf(abs(z))),
            "se": float(se), "n": n,
            "method": "sqrt(n)(S_n - F)/sqrt(F(1-F)) -> N(0,1) (Theorem 2.3.3)",
        }
    )


def cheatsheet():
    return "gb233: pointwise CLT for the EDF via the binomial count"
