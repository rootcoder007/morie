# morie.fn -- function file (rootcoder007/morie)
"""Kernel conditional quantile."""

import numpy as np

from ._horowitz import kernel, silverman_bw
from ._richresult import RichResult

__all__ = ["hrz_kernel_quantile", "horowitz_kernel_quantile_reg"]


def hrz_kernel_quantile(x, y, tau=0.5, grid=None, h=None, kernel_name="gaussian"):
    r"""Conditional quantile from a kernel conditional CDF (Horowitz
    Ch. 3):

    .. math:: \hat q_\tau(x) = \inf\{y : \hat F(y|x) \ge \tau\},
              \qquad \hat F(y|x) = \frac{\sum_i K_h(x-X_i)
              \mathbf 1\{Y_i \le y\}}{\sum_i K_h(x-X_i)}.

    Inverting a kernel-smoothed CDF rather than minimising a check
    loss. The estimate is automatically monotone in tau at each x --
    the crossing problem of independently fitted quantiles
    (:mod:`morie.fn.hrzsieqr`) cannot arise, because all quantiles
    come from one CDF.

    Parameters
    ----------
    x, y : array-like
        Regressor and response.
    tau : float in (0, 1) or array-like
        Quantile level(s).
    grid : array-like, optional
        Evaluation points.
    h : float, optional
        Bandwidth.
    kernel_name : str
        Kernel.

    Returns
    -------
    RichResult
        keys: ``grid``, ``quantile`` (len(grid) x len(tau)), ``tau``,
        ``bandwidth``, ``monotone_in_tau`` (True), ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 3 (conditional quantile estimation).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    taus = np.atleast_1d(np.asarray(tau, dtype=float))
    if np.any((taus <= 0) | (taus >= 1)):
        raise ValueError("tau values must lie in (0, 1).")
    h = silverman_bw(x) if h is None else float(h)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    g = np.linspace(x.min(), x.max(), 100) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    order = np.argsort(y)
    ys = y[order]
    out = np.empty((g.size, taus.size))
    for i, pt in enumerate(g):
        w = kernel((pt - x[order]) / h, kernel_name)
        tot = w.sum()
        if tot <= 0:
            out[i] = np.nan
            continue
        cdf = np.cumsum(w) / tot
        for j, t in enumerate(taus):
            k = int(np.searchsorted(cdf, t, side="left"))
            out[i, j] = ys[min(k, ys.size - 1)]
    return RichResult(payload={"grid": g,
                               "quantile": out[:, 0] if taus.size == 1 else out,
                               "tau": taus[0] if taus.size == 1 else taus,
                               "bandwidth": h, "monotone_in_tau": True,
                               "method": "Invert a kernel conditional CDF; no quantile crossing"})


def cheatsheet():
    return "hrzkqre: one CDF for all tau, so crossing cannot happen"


#: Catalogue alias for :func:`hrz_kernel_quantile`.
horowitz_kernel_quantile_reg = hrz_kernel_quantile
