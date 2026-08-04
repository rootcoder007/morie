# morie.fn -- function file (rootcoder007/morie)
"""Modified gamma kernel density estimator (Eq. 1.14)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["mgkde", "fauzi_modified_gamma_kde"]


def mgkde(x, grid, h):
    r"""Modified gamma kernel density estimator (Eq. 1.14).

    Eq. (1.14):

    .. math:: \tilde f_X(x) = [A_h(x)]^{2}\,[A_{4h}(x)]^{-1},

    with :math:`A_h` the raw gamma-kernel function (1.9) -- a sample mean
    of Gamma(:math:`h^{-1/2}`, :math:`x\sqrt h + h`) densities.

    The exponents 2 and -1 are the :math:`(t_1,t_2)` of Theorem 1.2, and
    the factor 4 on the bandwidth is fixed by that theorem too; neither is
    tunable, which is why they are not parameters here. The estimator is
    non-negative by construction, unlike the order-4 kernel alternative.

    The docstring of the backlog stub for this module gave the formula as
    a two-bandwidth geometric mean in a free parameter ``a``. That is the
    Chapter 2 DISTRIBUTION-function construction (2.5), not this one:
    Chapter 1 fixes the ratio at 4 because the expansion is in
    :math:`\sqrt h`, whereas Chapter 2 can carry a free ``a`` because its
    expansion is in :math:`h^2`.

    Parameters
    ----------
    x : array-like
        Sample on ``[0, infinity)``.
    grid : array-like
        Points at which to evaluate the density.
    h : float
        Bandwidth, ``h > 0``.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``ah``, ``a4h``, ``grid``, ``h``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eqs. (1.9) and (1.14).
    """
    from ._fauzi import agamma_kernel

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 1:
        raise ValueError("need at least one observation.")
    g = np.atleast_1d(np.asarray(grid, dtype=float))
    ah = np.asarray(agamma_kernel(xv, g, float(h)), dtype=float)
    a4h = np.asarray(agamma_kernel(xv, g, 4.0 * float(h)), dtype=float)
    if np.any(a4h <= 0):
        raise ValueError("A_4h vanished on the grid; (1.14) divides by it.")
    est = ah ** 2 / a4h
    return RichResult(
        payload={
            "estimate": [float(v) for v in est],
            "ah": [float(v) for v in ah],
            "a4h": [float(v) for v in a4h],
            "grid": [float(v) for v in g],
            "h": float(h),
            "n": int(n),
            "method": "modified gamma kernel density estimator (Eq. 1.14)",
        }
    )


fauzi_modified_gamma_kde = mgkde


def cheatsheet():
    return "fzmgkd: modified gamma KDE, A_h^2 / A_4h -- non-negative and boundary-free (Eq. 1.14)"


# CANONICAL TEST
# >>> r = mgkde([0.5, 1.0, 1.5, 2.0, 2.5], grid=[1.0], h=0.2)
# >>> abs(r['estimate'][0] - r['ah'][0] ** 2 / r['a4h'][0]) < 1e-15
# True
