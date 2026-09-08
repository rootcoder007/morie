# morie.fn -- function file (rootcoder007/morie)
"""Matheron's classical semivariogram estimator."""

from . import _array_core as np

from ._richresult import RichResult
from ._schaben import matheron

__all__ = ["schabenberger_matheron_estimator"]


def schabenberger_matheron_estimator(coords, z, lag_bins=None, cutoff=None,
                                     exact=False):
    r"""The classical semivariogram estimator, Schabenberger eq (4.24).

    .. math::
       \hat\gamma(h) = \frac{1}{2|N(h)|}\sum_{N(h)}
                       \{Z(s_i) - Z(s_j)\}^2

    Unbiased, even, zero at zero lag -- and acutely sensitive to a
    single extreme observation, because the squared difference
    magnifies it and the same observation contributes at several lags
    at once. Section 4.4.1 works this through on five points; that
    example is reproduced in the test suite.

    ``variance`` is the approximation (4.25),
    :math:`2\gamma(h)^2/|N(h)|`, which rises sharply at long range as
    the semivariogram approaches its sill while the number of available
    pairs collapses. ``n_pairs`` is returned per lag so the book's own
    guidance -- at least 30, preferably 50, pairs per class -- can be
    checked rather than assumed, and ``sparse_lags`` flags the classes
    that fall short.

    Parameters
    ----------
    coords : array-like, shape (n, d)
        Sampling locations.
    z : array-like, shape (n,)
        Observed values.
    lag_bins : int or sequence, optional
        Number of lag classes, or explicit bin edges.
    cutoff : float, optional
        Largest separation to use. Defaults to half the maximum, the
        book's recommendation on p. 155.
    exact : bool
        Treat every distinct separation as its own lag class. For
        small worked examples where the lags are dictated by the data.

    Returns
    -------
    RichResult
        ``lag``, ``gamma``, ``n_pairs``, ``variance``, ``sparse_lags``.

    References
    ----------
    Schabenberger and Gotway (2005), *Statistical Methods for Spatial
    Data Analysis*, section 4.4.1, equations (4.24) and (4.25),
    pp. 153-158. Matheron (1962).

    Examples
    --------
    >>> import numpy as np
    >>> co = np.array([[1, 1], [1, 4], [2, 2], [3, 1], [3, 4]], float)
    >>> z = np.array([1, 4, 2, 3, 20], float)
    >>> out = schabenberger_matheron_estimator(co, z, exact=True)
    >>> [round(float(g), 1) for g in out["gamma"]]
    [0.5, 65.0, 82.0, 74.5, 90.5]
    """
    lag, gam, npair, var = matheron(coords, z, lag_bins, cutoff, exact)
    sparse = [int(i) for i, k in enumerate(npair) if k < 30]
    return RichResult(
        payload={
            "estimate": gam,
            "empirical_variogram": gam,
            "gamma": gam,
            "lag": lag,
            "n_pairs": npair,
            "variance": var,
            "variance_note": (
                "equation (4.25), which assumes the squared differences are "
                "uncorrelated; it is a guide to precision, not an exact "
                "sampling variance"
            ),
            "sparse_lags": sparse,
            "sparse_note": (
                None if not sparse else
                "%d lag class(es) have fewer than 30 pairs; the book advises "
                "at least 30 and preferably 50 before reading a lag"
                % len(sparse)
            ),
            "n": int(np.asarray(z).size),
            "method": "Matheron classical semivariogram estimator",
        }
    )


def cheatsheet():
    return (
        "spmath: Matheron's semivariogram (4.24) with the (4.25) variance "
        "and a flag on lag classes too sparse to read"
    )
