# morie.fn -- function file (rootcoder007/morie)
"""Joint moment of EDF counts at two points."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_edf_joint_moment"]


def gibbons_edf_joint_moment(F_x, F_y, n):
    r"""Corollary 2.3.1.3: with :math:`T_n(x) = n S_n(x)` and x < y
    (so F(x) <= F(y)),

    .. math:: E[T_n(x) T_n(y)] = n F(x) + n(n - 1) F(x) F(y),

    hence the EDF covariance

    .. math:: \mathrm{Cov}[S_n(x), S_n(y)]
              = \frac{F(x)(1 - F(y))}{n},

    positive and Brownian-bridge shaped -- which is exactly why the
    K-S statistic's limit involves the bridge.

    Parameters
    ----------
    F_x, F_y : float in [0, 1]
        CDF values with F_x <= F_y.
    n : int
        Sample size.

    Returns
    -------
    RichResult
        keys: ``joint_moment``, ``cov_edf``, ``corr_edf``, ``n``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Corollary 2.3.1.3.
    """
    Fx, Fy = float(F_x), float(F_y)
    for nm, v in (("F_x", Fx), ("F_y", Fy)):
        if not 0 <= v <= 1:
            raise ValueError(f"{nm} must lie in [0, 1], got {v}.")
    if Fx > Fy:
        raise ValueError("need F_x <= F_y (x < y).")
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    jm = n * Fx + n * (n - 1) * Fx * Fy
    cov = Fx * (1 - Fy) / n
    vx = Fx * (1 - Fx) / n
    vy = Fy * (1 - Fy) / n
    corr = cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan
    return RichResult(
        payload={
            "joint_moment": float(jm), "cov_edf": float(cov),
            "corr_edf": float(corr), "n": n,
            "method": "E[T(x)T(y)] = nF(x) + n(n-1)F(x)F(y) (Corollary 2.3.1.3)",
        }
    )


def cheatsheet():
    return "gb2313: Cov[S(x),S(y)] = F(x)(1-F(y))/n -- the bridge covariance"
