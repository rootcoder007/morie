# morie.fn -- function file (rootcoder007/morie)
"""Durbin-Stuart / Daniels inequalities between tau and rho."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_tau_rho_relation"]


def gibbons_tau_rho_relation(tau, rho):
    r"""Check the exact inequalities linking Kendall's tau and
    Spearman's rho.

    For any bivariate distribution (Gibbons Ch. 11.4):

    .. math:: 3\tau - 1 \le 2\rho \le 1 + 2\tau
              \quad (\tau \ge 0),

    with the mirror-image bounds for negative tau, and always
    :math:`-1 \le 3\tau/2 - \rho/2 \le 1` (Daniels). The function
    evaluates whether an observed (tau, rho) pair is jointly
    attainable -- a diagnostic for reporting errors, since violating
    these bounds means at least one coefficient was miscomputed.

    Parameters
    ----------
    tau, rho : float in [-1, 1]
        Kendall and Spearman coefficients.

    Returns
    -------
    RichResult
        keys: ``consistent`` (bool), ``daniels_ok``,
        ``durbin_stuart_ok``, ``lower_2rho``, ``upper_2rho``, ``tau``,
        ``rho``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 11.4.

    Daniels, H. E. (1950). Rank correlation and population models.
    *Journal of the Royal Statistical Society B*, 12(2), 171-191.
    """
    tau = float(tau)
    rho = float(rho)
    for nm, v in (("tau", tau), ("rho", rho)):
        if not -1 <= v <= 1:
            raise ValueError(f"{nm} must lie in [-1, 1], got {v}.")
    # Daniels: -1 <= 3 tau - 2 rho <= 1
    daniels = -1 - 1e-12 <= 3 * tau - 2 * rho <= 1 + 1e-12
    # Durbin-Stuart, stated for tau >= 0; mirrored for tau < 0
    if tau >= 0:
        lo, hi = 3 * tau - 1, 1 + 2 * tau
    else:
        lo, hi = -1 + 2 * tau, 3 * tau + 1
    ds = lo - 1e-12 <= 2 * rho <= hi + 1e-12
    return RichResult(
        payload={
            "consistent": bool(daniels and ds), "daniels_ok": bool(daniels),
            "durbin_stuart_ok": bool(ds), "lower_2rho": float(lo),
            "upper_2rho": float(hi), "tau": tau, "rho": rho,
            "method": "tau/rho attainability bounds (Gibbons Ch. 11.4)",
        }
    )


def cheatsheet():
    return "gb1141: 3tau-1 <= 2rho <= 1+2tau; violation = a miscomputed coefficient"
