# morie.fn -- function file (rootcoder007/morie)
"""Bartlett kernel lag weights."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bartlett_kernel_weights"]


def bartlett_kernel_weights(lags, M=None):
    """Triangular lag weights that keep a long-run variance non-negative.

    Truncating an autocovariance sum at lag ``M`` with equal weights can
    produce a NEGATIVE variance estimate, which is not a rounding
    problem but a structural one.  The Bartlett taper fixes it: the
    triangular weights are the Fourier transform of a non-negative
    kernel, so the resulting estimator is positive semi-definite by
    construction.

    Formula: ``w_k = 1 - k / (M + 1)`` for ``k <= M``, and 0 beyond.

    Parameters
    ----------
    lags : int or array-like
        Number of lags, or the lag indices themselves.
    M : int, optional
        Bandwidth; the largest lag index by default.

    Returns
    -------
    RichResult
        ``w``, ``estimate`` (their sum), ``M``, ``n``.

    References
    ----------
    Newey, W. K. & West, K. D. (1987).  A simple, positive
    semi-definite, heteroskedasticity and autocorrelation consistent
    covariance matrix.  Econometrica 55:703-708, whose weights are
    those of Bartlett, M. S. (1950), Biometrika 37:1-16.
    """
    if isinstance(lags, (int, float)):
        ks = [float(k) for k in range(int(lags) + 1)]
    else:
        ks = C.vec(lags)
    Mv = float(M) if M is not None else max(ks)
    w = [max(1.0 - k / (Mv + 1.0), 0.0) for k in ks]
    return RichResult(payload={
        "w": w, "estimate": sum(w), "M": Mv, "n": len(w),
        "method": "Bartlett kernel lag weights"})


def cheatsheet():
    return "bartkw: Bartlett kernel lag weights."
