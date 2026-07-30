# morie.fn -- function file (rootcoder007/morie)
"""Kernel quantile estimator via kernel-smoothed empirical quantile function."""

import numpy as np

from ._richresult import RichResult

__all__ = ["fauzi_kernel_quantile", "fauzi_kernel_quantile_estimator"]


def fauzi_kernel_quantile(x, p, h=None):
    r"""Kernel quantile estimator (Fauzi Eq. 3.1):

    .. math:: \hat Q_{p,h} = \frac1h\int_0^1
              F_n^{-1}(u)\,K\!\left(\frac{u-p}{h}\right)du,

    which is a weighted sum of ORDER STATISTICS -- the book notes
    (3.1) can be rewritten that way, and that is how it is computed
    here.

    Smoothing in the PROBABILITY argument, not in x. The sample
    quantile uses a single order statistic and therefore jumps as
    :math:`p` crosses :math:`i/n`; this averages neighbouring order
    statistics with kernel weights, which removes the jumps and
    reduces variance. The book's motivation is explicit: in risk
    management the tails matter, and a single order statistic is a
    poor estimate out there.

    Parameters
    ----------
    x : array-like
        Sample.
    p : float or array-like
        Probability levels in (0, 1).
    h : float, optional
        Bandwidth on the probability scale.

    Returns
    -------
    RichResult
        keys: ``p``, ``quantile``, ``sample_quantile``, ``bandwidth``,
        ``weights_sum``, ``smooths_in``, ``n``, ``method``.
    References
    ----------
    Fauzi and Maesono (2023), Eq. (3.1). From the PDF.
    """
    from ._fauzi import kernel_W

    xv = np.sort(np.asarray(x, dtype=float).ravel())
    n = xv.size
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}.")
    pv = np.atleast_1d(np.asarray(p, dtype=float)).ravel()
    if np.any((pv <= 0) | (pv >= 1)):
        raise ValueError("probability levels must lie strictly in (0, 1).")
    hh = float(n ** -0.4) if h is None else float(h)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")
    # Weight on the i-th order statistic is the kernel mass of the
    # interval ((i-1)/n, i/n], which is an EXACT difference of the
    # integrated kernel -- no quadrature. A numerical integral over a
    # fixed u-grid silently fails here: once n exceeds the grid
    # resolution each bin holds one point or none, and a one-point
    # trapezoid is zero, so nearly every weight vanishes and the
    # estimate collapses onto whichever order statistic survives.
    edges = np.arange(n + 1) / n
    Wl = kernel_W((edges[:, None] - pv[None, :]) / hh)
    wi = Wl[1:] - Wl[:-1]
    wsum = wi.sum(axis=0)
    q = (wi * xv[:, None]).sum(axis=0) / wsum
    return RichResult(payload={
        "p": pv, "quantile": q,
        "sample_quantile": np.quantile(xv, pv),
        "bandwidth": hh, "weights_sum": wsum,
        "smooths_in": "the PROBABILITY argument, not in x",
        "why": "the sample quantile uses one order statistic and jumps as p "
               "crosses i/n; the tails are exactly where that hurts",
        "n": int(n),
        "method": "Kernel quantile estimator (3.1) as a weighted sum of order statistics"})


def cheatsheet():
    return "fzkqe: smooths in p, not in x -- averages neighbouring order statistics"


#: Catalogue alias for :func:`fauzi_kernel_quantile`.
fauzi_kernel_quantile_estimator = fauzi_kernel_quantile
