# morie.fn -- function file (rootcoder007/morie)
"""Estimation error in vector form."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_estimation_error_vector_form"]


def rangayyan_ch3_estimation_error_vector_form(d, w, x, n=None):
    r"""Estimation error in vector form (Rangayyan Ch. 3):

    .. math:: e(n) = d(n) - \mathbf{w}^T \mathbf{x}(n),

    with w the tap-weight vector and x(n) the tap-input vector. This
    is the same error as :mod:`morie.fn.rng137` once the estimate is
    written as an inner product -- which is the step that makes the
    gradient computable in closed form.

    Parameters
    ----------
    d : array-like, shape (N,)
        Desired response.
    w : array-like, shape (p,)
        Tap weights.
    x : array-like, shape (N, p)
        Tap-input vectors, one row per time index.
    n : int, optional
        Index to report.

    Returns
    -------
    RichResult
        keys: ``error``, ``error_at_n``, ``mse``, ``estimate``,
        ``N``, ``p``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (adaptive filters).
    """
    d = np.asarray(d, dtype=float).ravel()
    w = np.atleast_1d(np.asarray(w, dtype=float)).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != d.size:
        X = X.T
    if X.shape[0] != d.size:
        raise ValueError("x must have one row per entry of d.")
    if X.shape[1] != w.size:
        raise ValueError(f"x has {X.shape[1]} columns but w has {w.size} weights.")
    est = X @ w
    e = d - est
    at_n = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < e.size:
            raise ValueError(f"n must lie in 0..{e.size - 1}, got {idx}.")
        at_n = float(e[idx])
    return RichResult(payload={"error": e, "error_at_n": at_n,
                               "mse": float(np.mean(e**2)), "estimate": est,
                               "N": int(d.size), "p": int(w.size),
                               "method": "e(n) = d(n) - w^T x(n)"})


def cheatsheet():
    return "rng140: inner-product form is what makes the gradient closed-form"
