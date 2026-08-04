# morie.fn -- function file (rootcoder007/morie)
"""Boundary-free kernel distribution function estimator (Eq. 5.5)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["bfkdf", "fauzi_bdfree_kdfe_test"]


def bfkdf(x, grid=None, ginv=None, h=None):
    r"""Boundary-free kernel distribution function estimator (Eq. 5.5).

    Eq. (5.5):

    .. math:: \tilde F_X(x) = \frac1n\sum_{i=1}^n
              W\!\Big(\frac{g^{-1}(x) - g^{-1}(X_i)}h\Big),
              \qquad x \in \Omega.

    It looks like nothing more than substituting :math:`g^{-1}` into the
    naive estimator, and that is exactly what it is. The reason it WORKS
    is the change-of-variable property of a distribution function: if
    :math:`Y = g^{-1}(X)` then :math:`F_X(x) = F_Y(g^{-1}(x))` exactly,
    for an increasing :math:`g`. No Jacobian appears.

    That is why the trick is available here and not to a density
    estimator, where the same substitution needs the
    :math:`1/g'(g^{-1}(x))` factor of (5.9). Sec. 5.2 makes the point
    explicitly: the property "cannot always be done to other
    probability-related functions".

    Estimating on the transformed scale, where the support is the whole
    line, means a symmetric kernel puts no mass outside :math:`\Omega`
    after mapping back -- so the O(h) boundary bias simply never arises,
    and Theorem 5.2 gets O(h^2) everywhere including at the edge.

    Parameters
    ----------
    x : array-like
        Sample, inside the support ``Omega``.
    grid : array-like, optional
        Evaluation points; defaults to the sorted sample.
    ginv : callable, optional
        :math:`g^{-1}`; defaults to ``log``, i.e. ``g = exp`` and
        ``Omega = (0, infinity)``.
    h : float, optional
        Bandwidth on the TRANSFORMED scale; defaults to the
        distribution-function rule applied to ``ginv(x)``.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``grid``, ``h``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eq. (5.5).
    """
    from . import _stats_core as stats
    from ._fauzi import kdfe_bandwidth

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 2:
        raise ValueError(f"need at least two observations, got {n}.")
    if ginv is None:
        if np.any(xv <= 0):
            raise ValueError("the default g = exp needs data on (0, infinity).")
        ginv = np.log
    y = np.asarray([float(ginv(float(t))) for t in xv], dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("g^-1 left the real line on some observation.")
    if h is None:
        h = kdfe_bandwidth(y)
    h = float(h)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    g = np.sort(xv) if grid is None else np.atleast_1d(np.asarray(grid, dtype=float))
    est = np.asarray(
        [float(np.mean(stats.norm.cdf((float(ginv(float(t))) - y) / h))) for t in g],
        dtype=float,
    )
    return RichResult(
        payload={
            "estimate": [float(v) for v in est],
            "grid": [float(v) for v in g],
            "h": h,
            "n": int(n),
            "method": "boundary-free kernel distribution function estimator (Eq. 5.5)",
        }
    )


fauzi_bdfree_kdfe_test = bfkdf


def cheatsheet():
    return "fzbfkf: boundary-free KDFE: substitute g^-1, no Jacobian, because F_X(x) = F_Y(g^-1(x)) exactly"


# CANONICAL TEST
# >>> r = bfkdf([0.5, 1.0, 1.5, 2.0, 3.0], grid=[1.5])
# >>> 0.0 <= r['estimate'][0] <= 1.0
# True
