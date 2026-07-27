# morie.fn -- function file (rootcoder007/morie)
"""Threshold (truncated) jump-robust realised variance."""

import numpy as np

from ._richresult import RichResult

__all__ = ["vol_jump_robust_var"]


def vol_jump_robust_var(r_intraday, threshold=None, c=3.0):
    r"""Truncated realised variance.

    .. math:: RV_J = \sum_i r_i^2\, \mathbb{1}\{|r_i| \le \theta\},

    discarding returns whose magnitude exceeds the threshold --
    Mancini's truncation approach to separating diffusion from jumps.
    The default data-driven threshold is :math:`\theta = c \cdot
    \sqrt{BPV/m}` (c local standard deviations under the bipower
    proxy), so a jump several sigma wide is excluded from the
    continuous part.

    Parameters
    ----------
    r_intraday : array-like, shape (m,)
        Intraday returns.
    threshold : float, optional
        Absolute cutoff; overrides the data-driven default.
    c : float, default 3.0
        Multiplier for the default threshold.

    Returns
    -------
    RichResult
        keys: ``rv_truncated``, ``rv`` (untruncated), ``threshold``,
        ``n_excluded``, ``n_returns``, ``method``.

    References
    ----------
    Mancini, C. (2009). Non-parametric threshold estimation for
    models with stochastic diffusion coefficient and jumps.
    *Scandinavian Journal of Statistics*, 36(2), 270-296.

    Barndorff-Nielsen, O. E. & Shephard, N. (2004). *Journal of
    Financial Econometrics*, 2(1), 1-48. (the bipower local-variance
    proxy)
    """
    r = np.asarray(r_intraday, dtype=float).ravel()
    m = r.size
    if m < 3:
        raise ValueError("need at least 3 intraday returns.")
    if threshold is None:
        bpv = (np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
        threshold = float(c) * np.sqrt(bpv / m)
    threshold = float(threshold)
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}.")

    keep = np.abs(r) <= threshold
    return RichResult(
        payload={
            "rv_truncated": float((r[keep] ** 2).sum()),
            "rv": float((r**2).sum()),
            "threshold": threshold,
            "n_excluded": int((~keep).sum()),
            "n_returns": int(m),
            "method": "Threshold-truncated realised variance",
        }
    )


def cheatsheet():
    return "voljr: sum r^2 over |r| <= theta; default theta = c sqrt(BPV/m)"
