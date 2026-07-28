# morie.fn -- function file (rootcoder007/morie)
"""Probability density estimate."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_pdf_estimate"]


def rangayyan_pdf_estimate(x, bins=None, bw=None, method="kde", grid=None):
    r"""Probability density estimate (Rangayyan Ch. 3), by histogram
    or Gaussian kernel:

    .. math:: \hat p(x) = \frac{1}{Nh}\sum_i
              K\!\left(\frac{x - x_i}{h}\right),
              \qquad K = \text{Gaussian}.

    The bandwidth h controls a bias-variance trade the bin count
    cannot express as smoothly, which is why the kernel form is the
    default. With ``bw`` omitted, Silverman's rule
    :math:`h = 0.9\,\min(\sigma, IQR/1.34)\,N^{-1/5}` is used and
    reported, so the choice is visible rather than buried.

    Parameters
    ----------
    x : array-like
        Samples.
    bins : int, optional
        Histogram bins (method="hist").
    bw : float, optional
        Kernel bandwidth.
    method : {"kde", "hist"}
        Estimator.
    grid : array-like, optional
        Evaluation points.

    Returns
    -------
    RichResult
        keys: ``grid``, ``density``, ``bandwidth``, ``integrates_to``,
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (probability density estimation).
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("x must have at least 2 samples.")
    g = np.linspace(x.min() - 3 * x.std(), x.max() + 3 * x.std(), 512) \
        if grid is None else np.asarray(grid, dtype=float).ravel()
    if method == "hist":
        nb = int(bins) if bins is not None else max(5, int(np.sqrt(x.size)))
        if nb < 1:
            raise ValueError("bins must be positive.")
        counts, edges = np.histogram(x, bins=nb, density=True)
        centres = 0.5 * (edges[1:] + edges[:-1])
        dens = np.interp(g, centres, counts, left=0.0, right=0.0)
        h = float(edges[1] - edges[0])
    elif method == "kde":
        if bw is None:
            iqr = float(np.subtract(*np.percentile(x, [75, 25])))
            spread = min(float(x.std()), iqr / 1.34) if iqr > 0 else float(x.std())
            h = 0.9 * max(spread, 1e-12) * x.size ** (-0.2)
        else:
            h = float(bw)
        if h <= 0:
            raise ValueError(f"bandwidth must be positive, got {h}.")
        z = (g[:, None] - x[None, :]) / h
        dens = np.exp(-0.5 * z**2).sum(axis=1) / (x.size * h * np.sqrt(2 * np.pi))
    else:
        raise ValueError("method must be 'kde' or 'hist'.")
    return RichResult(payload={"grid": g, "density": dens, "bandwidth": h,
                               "integrates_to": float(np.trapezoid(dens, g)),
                               "method": f"{method} density; Silverman bandwidth when unset"})


def cheatsheet():
    return "rgpdfest: Silverman h reported, not hidden; integral returned as a check"
