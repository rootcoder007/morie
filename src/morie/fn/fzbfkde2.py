# morie.fn -- function file (rootcoder007/morie)
"""Boundary-free kernel density estimator (Eq. 5.9)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["bfkde", "fauzi_bdfree_density_from_cdf"]


def bfkde(x, grid=None, ginv=None, dg=None, h=None):
    r"""Boundary-free kernel density estimator (Eq. 5.9).

    Eq. (5.9):

    .. math:: \tilde f_X(x) = \frac1{nhg'(g^{-1}(x))}\sum_{i=1}^n
              K\!\Big(\frac{g^{-1}(x)-g^{-1}(X_i)}h\Big),
              \qquad x\in\Omega.

    The :math:`1/g'(g^{-1}(x))` factor is the Jacobian, and it is exactly
    what the distribution-function estimator (5.5) does NOT need. A
    density is a derivative, so it transforms with a Jacobian; a
    distribution function is a probability, so it does not. That one
    difference runs through the whole chapter.

    Bias and variance are Theorem 5.5:
    :math:`h^2c_2(x)\mu_2(K)/(2g'(g^{-1}(x)))` and
    :math:`f_X(x)\int K^2/(nhg'(g^{-1}(x)))`. Note the variance is
    :math:`O(1/(nh))`, not the :math:`O(h/n)` of the distribution
    estimators -- so THIS estimator takes the density bandwidth rate
    :math:`n^{-1/5}`, and the default here is Silverman's rule on the
    transformed scale, not the cube-root rule the rest of the suite uses.

    Parameters
    ----------
    x : array-like
        Sample, inside the support ``Omega``.
    grid : array-like, optional
        Evaluation points; defaults to the sorted sample.
    ginv : callable, optional
        :math:`g^{-1}`; defaults to ``log``.
    dg : callable, optional
        :math:`g'` as a function of ``g^{-1}(x)``; defaults to ``exp``,
        the derivative matching the default ``ginv = log``.
    h : float, optional
        Bandwidth on the transformed scale; defaults to Silverman's
        ``n^(-1/5)`` rule, because this is a DENSITY estimator.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``grid``, ``h``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eq. (5.9), Theorem 5.5.
    """
    from . import _stats_core as stats

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 2:
        raise ValueError(f"need at least two observations, got {n}.")
    if ginv is None:
        if np.any(xv <= 0):
            raise ValueError("the default g = exp needs data on (0, infinity).")
        ginv = np.log
        if dg is None:
            dg = np.exp
    if dg is None:
        raise ValueError("supply dg alongside a custom ginv.")
    y = np.asarray([float(ginv(float(t))) for t in xv], dtype=float)
    if h is None:
        sd = float(np.std(y, ddof=1))
        if sd <= 0:
            sd = 1.0
        h = 1.06 * sd * n ** (-0.2)
    h = float(h)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    g = np.sort(xv) if grid is None else np.atleast_1d(np.asarray(grid, dtype=float))
    out = np.empty(g.size)
    for i, t in enumerate(g):
        z = float(ginv(float(t)))
        jac = float(dg(z))
        if jac <= 0:
            raise ValueError("g' must be positive; g is an increasing bijection (D4).")
        out[i] = float(np.mean(stats.norm.pdf((z - y) / h))) / (h * jac)
    return RichResult(
        payload={
            "estimate": [float(v) for v in out],
            "grid": [float(v) for v in g],
            "h": h,
            "n": int(n),
            "method": "boundary-free kernel density estimator (Eq. 5.9)",
        }
    )


fauzi_bdfree_density_from_cdf = bfkde


def cheatsheet():
    return "fzbfkde2: boundary-free KDE -- needs the 1/g' Jacobian the df estimator (5.5) does not"


# CANONICAL TEST
# >>> r = bfkde([0.5, 1.0, 1.5, 2.0, 3.0], grid=[1.5])
# >>> r['estimate'][0] > 0
# True
