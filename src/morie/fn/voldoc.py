# morie.fn -- function file (rootcoder007/morie)
"""Continuous/jump decomposition of realised variance (BNS)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["vol_decomposed_realised"]


def vol_decomposed_realised(RV, BPV):
    r"""Split realised variance into continuous and jump parts.

    .. math:: J_d = \max(RV_d - BPV_d, 0), \qquad C_d = RV_d - J_d,

    using bipower variation's jump robustness: BPV estimates the
    integrated (continuous) variance, so any excess of RV over it is
    attributed to jumps, truncated at zero because sampling noise
    makes the raw difference go negative on jump-free days.

    Parameters
    ----------
    RV : array-like
        Realised variance per day.
    BPV : array-like
        Bipower variation per day (same length).

    Returns
    -------
    RichResult
        keys: ``continuous``, ``jump``, ``jump_share`` (of total RV),
        ``n_days``, ``method``.

    References
    ----------
    Barndorff-Nielsen, O. E. & Shephard, N. (2004). Power and bipower
    variation with stochastic volatility and jumps. *Journal of
    Financial Econometrics*, 2(1), 1-37.
    """
    rv = np.atleast_1d(np.asarray(RV, dtype=float))
    bpv = np.atleast_1d(np.asarray(BPV, dtype=float))
    if rv.shape != bpv.shape:
        raise ValueError("RV and BPV must have the same length.")
    if np.any(rv < 0) or np.any(bpv < 0):
        raise ValueError("variances cannot be negative.")

    jump = np.maximum(rv - bpv, 0.0)
    cont = rv - jump
    tot = rv.sum()
    scalar = np.ndim(RV) == 0
    return RichResult(
        payload={
            "continuous": float(cont[0]) if scalar else cont,
            "jump": float(jump[0]) if scalar else jump,
            "jump_share": float(jump.sum() / tot) if tot > 0 else 0.0,
            "n_days": int(rv.size),
            "method": "BNS decomposition: J = max(RV - BPV, 0), C = RV - J",
        }
    )


def cheatsheet():
    return "voldoc: J = max(RV - BPV, 0); C = RV - J (BNS 2004)"
