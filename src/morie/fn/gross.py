# morie.fn -- tail3 batch (rootcoder007/morie)
"""Gross-error sensitivity of an estimator.

Source consulted: Hampel, F.R., Ronchetti, E.M., Rousseeuw, P.J. & Stahel,
W.A. (1986). *Robust Statistics: The Approach Based on Influence Functions*.
Wiley, section 2.1c.  The gross-error sensitivity of a functional ``T`` at
``F`` is

    gamma* = sup_x |IF(x; T, F)|

the worst influence a small amount of contamination can have on the value of
the estimator.  ``T`` is B-robust at ``F`` exactly when gamma* is finite.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gross_error_sensitivity"]


def gross_error_sensitivity(IF, x=None):
    """Gross-error sensitivity gamma* = sup_x |IF(x)|.

    Parameters
    ----------
    IF : array-like
        Influence-function values IF(x_i; T, F) on a grid.
    x : array-like, optional
        Grid the influence function was evaluated on; used only to report the
        argument at which the supremum is attained.

    Returns
    -------
    RichResult
        estimate (gamma*), gamma_star, xmax, imax, brobust, n, method.

    References
    ----------
    Hampel, Ronchetti, Rousseeuw & Stahel (1986), section 2.1c.
    """
    IF = np.atleast_1d(np.asarray(IF, dtype=float)).ravel()
    n = int(IF.size)
    a = np.abs(IF)
    imax = int(np.argmax(a))
    gamma = float(a[imax])
    if x is None:
        xmax = float(imax)
    else:
        xg = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        xmax = float(xg[imax])
    return RichResult(
        payload={
            "estimate": gamma,
            "gamma_star": gamma,
            "xmax": xmax,
            "imax": imax,
            "brobust": bool(gamma < float("inf")),
            "n": n,
            "method": "Gross-error sensitivity (Hampel et al. 1986)",
        }
    )


# CANONICAL TEST
# >>> r = gross_error_sensitivity([0.1, -0.9, 0.4], x=[1.0, 2.0, 3.0])
# >>> assert abs(r["estimate"] - 0.9) < 1e-12
# >>> assert abs(r["xmax"] - 2.0) < 1e-12


def cheatsheet():
    return "gross(IF, x): gross-error sensitivity gamma* = sup|IF|."
