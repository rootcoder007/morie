# morie.fn -- function file (rootcoder007/morie)
"""Moran's I global spatial autocorrelation (Moran 1950)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["morans_i"]


def morans_i(x, W):
    r"""Global spatial autocorrelation of ``x`` under the connectivity ``W``.

    .. math::

        I = \frac{n}{(n-1)S^2 w_{\cdot\cdot}}
            \sum_{i=1}^{n}\sum_{j=1}^{n} w_{ij}
            (Z(s_i)-\bar{Z})(Z(s_j)-\bar{Z})

    with :math:`S^2 = (n-1)^{-1}\sum_i (Z(s_i)-\bar{Z})^2` and
    :math:`w_{\cdot\cdot} = \sum_{i,j} w_{ij}`. Since
    :math:`(n-1)S^2 = \sum_i (Z(s_i)-\bar{Z})^2`, this is the familiar form

    .. math::

        I = \frac{n}{w_{\cdot\cdot}}\,
            \frac{\sum_i\sum_j w_{ij}(x_i-\bar{x})(x_j-\bar{x})}
                 {\sum_i (x_i-\bar{x})^2}.

    Parameters
    ----------
    x : array-like, shape (n,)
        Attribute observed at the ``n`` sites.
    W : array-like, shape (n, n)
        Spatial connectivity weights :math:`w_{ij}`. The diagonal is ignored
        (a site is not its own neighbour) and is zeroed before use.

    Returns
    -------
    RichResult
        keys: ``estimate`` (:math:`I`), ``expected`` (:math:`E[I]`),
        ``n``, ``W_sum`` (:math:`w_{\cdot\cdot}`), ``method``.

    Raises
    ------
    ValueError
        If shapes disagree, if ``n < 2``, if all weights are zero, or if
        ``x`` is constant -- the denominator is then zero and :math:`I` is
        undefined, not zero.

    References
    ----------
    Schabenberger, O., & Gotway, C. A. *Statistical Methods for Spatial Data
        Analysis*. Eq. (1.14), p. 21, Section 1.3.2.2 ("The Geary and Moran
        Statistics"), attributed to Moran (1950). The null expectation
        :math:`E[I] = -1/(n-1)` is on p. 22, where the book also states the
        interpretation: :math:`I > E[I]` means a site "tends to be connected
        to sites that have similar attribute values".
    Moran, P. A. P. (1950). Notes on continuous stochastic phenomena.
        *Biometrika*, 37(1/2), 17-23.

    Notes
    -----
    :math:`E[I] = -1/(n-1)`, **not zero**. Comparing :math:`I` against 0
    rather than against :math:`E[I]` biases every conclusion toward "positive
    autocorrelation", and the bias grows as :math:`n` shrinks -- at
    :math:`n = 10` the null is already :math:`-0.111`. The expectation is
    returned alongside the estimate so the comparison is available without
    the caller having to know this.

    The book is emphatic that "the assumptions of constant mean and constant
    variance ... must not be taken lightly": its Example 1.7 puts *independent*
    draws on a lattice with a non-constant mean function and obtains
    :math:`I = 0.2597` with :math:`p = 0.00011`. There is no spatial
    autocorrelation in those data at all -- the statistic is detecting the
    mean trend. :math:`I` alone cannot tell the two apart.
    """
    z = np.asarray(x, dtype=float).ravel()
    w = np.asarray(W, dtype=float)
    n = z.size
    if n < 2:
        raise ValueError(f"need at least 2 sites; got {n}")
    if w.shape != (n, n):
        raise ValueError(f"W must have shape ({n}, {n}) to match x; got {w.shape}")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(w)):
        raise ValueError("x and W must be finite")
    # A site is not its own neighbour; self-weights would inflate I toward 1.
    w = w - np.diag(np.diag(w))
    w_sum = float(w.sum())
    if w_sum == 0.0:
        raise ValueError(
            "all off-diagonal weights are zero, so w.. = 0 and Eq. (1.14) is "
            "undefined. Check that W encodes at least one neighbour pair."
        )
    dev = z - z.mean()
    denom = float(dev @ dev)
    if denom == 0.0:
        raise ValueError(
            "x is constant, so sum (x_i - xbar)^2 = 0 and I is undefined. "
            "A constant surface has no variation to be autocorrelated."
        )
    numer = float(dev @ w @ dev)
    I = (n / w_sum) * (numer / denom)
    return RichResult(
        payload={
            "estimate": I,
            "expected": -1.0 / (n - 1),
            "n": int(n),
            "W_sum": w_sum,
            "method": "Moran's I global spatial autocorrelation (Eq 1.14, Moran 1950)",
        }
    )


def cheatsheet():
    return "morisp: Moran's I = (n/w..) * z'Wz / z'z, E[I] = -1/(n-1) (Moran 1950)."
