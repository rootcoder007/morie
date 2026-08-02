# morie.fn -- function file (rootcoder007/morie)
"""Cross-functional correlation between paired curves (Ramsay & Silverman 2005)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["functional_correlation"]


def functional_correlation(X, Y, argvals=None):
    r"""Correlation between two samples of curves observed on a shared grid.

    Centring each sample by its own **mean function** and writing the
    :math:`L^2` inner product as an integral over the argument,

    .. math::

        r = \frac{\int \operatorname{cov}\{X(t), Y(t)\}\,dt}
                 {\sqrt{\int \operatorname{var}\{X(t)\}\,dt \;
                        \int \operatorname{var}\{Y(t)\}\,dt}}

    Parameters
    ----------
    X, Y : array-like, shape (n_curves, n_points)
        Paired samples of curves, evaluated on a common grid. Row ``i`` of
        ``X`` is paired with row ``i`` of ``Y``.
    argvals : array-like, shape (n_points,), optional
        The grid :math:`t`. Defaults to equally spaced points, in which case
        the spacing cancels and the result is the discrete inner-product
        form. Supply it when the grid is **irregular**: with unequal spacing
        an unweighted sum silently overweights the densely sampled region.

    Returns
    -------
    RichResult
        keys: ``estimate`` (:math:`r`), ``cov_integral``, ``var_x_integral``,
        ``var_y_integral``, ``n_curves``, ``n_points``, ``method``.

    Raises
    ------
    ValueError
        If ``X`` and ``Y`` have different shapes, if there are fewer than 2
        curves, if ``argvals`` does not match the grid, or if either sample
        has zero total variation.

    References
    ----------
    Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data Analysis*,
        2nd ed. Springer.

    Notes
    -----
    This is a **single scalar summarising the whole pair of samples**, not a
    pointwise correlation curve :math:`r(t)`. The two answer different
    questions and the distinction is the usual confusion here: a pair of
    samples can have :math:`r(t)` near 1 at every :math:`t` and a modest
    integrated :math:`r` if the variance is concentrated where the curves
    disagree.

    Integration uses the trapezoid rule over ``argvals``. Centring is by the
    sample mean function, so at ``n_curves = 2`` the two centred curves are
    exact negatives of each other and :math:`r` is forced to :math:`\pm 1`
    regardless of the data -- meaningful use needs more curves than that.
    """
    xa = np.asarray(X, dtype=float)
    ya = np.asarray(Y, dtype=float)
    if xa.ndim != 2 or ya.ndim != 2:
        raise ValueError(
            f"X and Y must be 2-D (n_curves x n_points); got {xa.shape} and {ya.shape}"
        )
    if xa.shape != ya.shape:
        raise ValueError(
            f"X and Y must have the same shape -- the curves are paired; got "
            f"{xa.shape} and {ya.shape}"
        )
    n_curves, n_points = xa.shape
    if n_curves < 2:
        raise ValueError(f"need at least 2 curves to have any variation; got {n_curves}")
    if n_points < 2:
        raise ValueError(f"need at least 2 grid points to integrate; got {n_points}")
    if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
        raise ValueError("X and Y must be finite")
    if argvals is None:
        t = np.linspace(0.0, 1.0, n_points)
    else:
        t = np.asarray(argvals, dtype=float).ravel()
        if t.size != n_points:
            raise ValueError(f"argvals must have length {n_points}; got {t.size}")
        if np.any(np.diff(t) <= 0):
            raise ValueError("argvals must be strictly increasing")
    xc = xa - xa.mean(axis=0, keepdims=True)
    yc = ya - ya.mean(axis=0, keepdims=True)
    # Pointwise (co)variances with the same denominator, which then cancels.
    cov_t = (xc * yc).sum(axis=0) / (n_curves - 1)
    var_x_t = (xc * xc).sum(axis=0) / (n_curves - 1)
    var_y_t = (yc * yc).sum(axis=0) / (n_curves - 1)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    c = float(trapezoid(cov_t, t))
    vx = float(trapezoid(var_x_t, t))
    vy = float(trapezoid(var_y_t, t))
    if vx <= 0 or vy <= 0:
        raise ValueError(
            "one of the samples has zero integrated variance -- every curve is "
            "identical, so no correlation is defined."
        )
    r = c / np.sqrt(vx * vy)
    return RichResult(
        payload={
            "estimate": float(r),
            "cov_integral": c,
            "var_x_integral": vx,
            "var_y_integral": vy,
            "n_curves": int(n_curves),
            "n_points": int(n_points),
            "method": "integrated functional correlation (Ramsay & Silverman 2005)",
        }
    )


def cheatsheet():
    return "fpccor: r = int cov(X,Y) dt / sqrt(int var X dt * int var Y dt) (Ramsay & Silverman 2005)."
