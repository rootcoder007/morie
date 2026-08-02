# morie.fn -- function file (rootcoder007/morie)
"""Standard kernel density estimator (Rosenblatt-Parzen)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["fauzi_kde", "fauzi_standard_kde"]


def fauzi_kde(x, grid=None, h=None):
    r"""Rosenblatt-Parzen kernel density estimator (Fauzi Ch. 1):

    .. math:: \hat f_h(x) = \frac{1}{nh}\sum_{i=1}^{n}
              K\!\left(\frac{x - X_i}{h}\right).

    The baseline the whole book is written against. Its interior bias
    is :math:`h^2\mu_2(K)f''(x)/2 + o(h^2)` and its variance
    :math:`R(K)f(x)/(nh)`, so bias and variance pull the bandwidth in
    opposite directions -- the trade that :mod:`morie.fn.fzmise`
    resolves.

    Its failure is at a BOUNDED support's edge. A symmetric kernel
    placed near a boundary puts mass outside the support, and the
    bias there is O(h), not O(h^2): it does not improve at the same
    rate as h shrinks, so the estimator is inconsistent at the
    boundary point itself. That single fact motivates the gamma
    kernel of :mod:`morie.fn.fzgkde` and the transformations of
    Chapter 4, and ``boundary_consistent`` records it.

    Parameters
    ----------
    x : array-like
        Sample.
    grid : array-like, optional
        Evaluation points.
    h : float, optional
        Bandwidth; Silverman's rule otherwise.

    Returns
    -------
    RichResult
        keys: ``grid``, ``density``, ``bandwidth``, ``mass``,
        ``interior_bias_order``, ``boundary_bias_order``,
        ``boundary_consistent`` (False), ``n``, ``method``.
    References
    ----------
    Fauzi, R. R. and Maesono, Y. *Statistical Inference Based on
    Kernel Distribution Function Estimators*. Springer, Ch. 1.
    """
    from ._fauzi import kernel_K

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    if h is None:
        sd = float(np.std(xv, ddof=1))
        iqr = float(np.subtract(*np.percentile(xv, [75, 25])))
        scale = min(sd, iqr / 1.349) if iqr > 0 else sd
        hh = 1.06 * (scale if scale > 0 else 1.0) * n ** -0.2
    else:
        hh = float(h)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")
    g = np.linspace(xv.min() - 3 * hh, xv.max() + 3 * hh, 200) \
        if grid is None else np.atleast_1d(np.asarray(grid, dtype=float))
    dens = kernel_K((g[:, None] - xv[None, :]) / hh).sum(axis=1) / (n * hh)
    return RichResult(payload={
        "grid": g, "density": dens, "bandwidth": hh,
        "mass": float(np.trapezoid(dens, g)),
        "interior_bias_order": "O(h^2)",
        "boundary_bias_order": "O(h) -- does NOT vanish at the same rate",
        "boundary_consistent": False,
        "n": int(n),
        "method": "Rosenblatt-Parzen KDE; the boundary failure is what Ch. 1 and Ch. 4 are for"})


def cheatsheet():
    return "fzkde: O(h^2) inside, O(h) at a boundary -- inconsistent exactly at the edge"


#: Catalogue alias for :func:`fauzi_kde`.
fauzi_standard_kde = fauzi_kde
