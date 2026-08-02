# morie.fn -- function file (rootcoder007/morie)
"""Realised power variation of order p."""

from . import _array_core as np
from scipy import special

from ._richresult import RichResult

__all__ = ["vol_power_variation"]


def vol_power_variation(r_intraday, p=1.0):
    r"""Realised power variation.

    .. math:: PV_p = \sum_i |r_i|^p,

    with the standardised version :math:`\mu_p^{-1} m^{p/2 - 1} PV_p`,
    where :math:`\mu_p = E|Z|^p = 2^{p/2}\,\Gamma(\tfrac{p+1}{2})/
    \Gamma(\tfrac12)` -- BNS's scaling that makes the p = 2 case
    coincide with realised variance and p < 2 cases jump-robust.

    Parameters
    ----------
    r_intraday : array-like, shape (m,)
        Intraday returns.
    p : float > 0, default 1.0
        Power order.

    Returns
    -------
    RichResult
        keys: ``pv`` (raw sum), ``pv_standardised``, ``mu_p``, ``p``,
        ``n_returns``, ``method``.

    References
    ----------
    Barndorff-Nielsen, O. E. & Shephard, N. (2004). Power and bipower
    variation with stochastic volatility and jumps. *Journal of
    Financial Econometrics*, 2(1), 1-48.
    """
    r = np.asarray(r_intraday, dtype=float).ravel()
    m = r.size
    if m < 2:
        raise ValueError("need at least 2 intraday returns.")
    p = float(p)
    if p <= 0:
        raise ValueError(f"p must be positive, got {p}.")

    pv = float((np.abs(r) ** p).sum())
    mu_p = 2 ** (p / 2) * special.gamma((p + 1) / 2) / special.gamma(0.5)
    std = pv * m ** (p / 2 - 1) / mu_p

    return RichResult(
        payload={
            "pv": pv,
            "pv_standardised": float(std),
            "mu_p": float(mu_p),
            "p": p,
            "n_returns": int(m),
            "method": f"Realised power variation (p = {p:g})",
        }
    )


def cheatsheet():
    return "volpow: PV_p = sum |r|^p; standardised by mu_p and m^(p/2-1) (BNS 2004)"
