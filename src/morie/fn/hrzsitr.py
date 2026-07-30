# morie.fn -- function file (rootcoder007/morie)
"""Sieve nonparametric IV."""

import numpy as np

from ._richresult import RichResult

__all__ = ["hrz_sieve_iv", "horowitz_sieve_npiv"]


def hrz_sieve_iv(T, Ey_w, K=None):
    r"""Sieve (series) solution of the nonparametric IV equation
    (Horowitz Sec. 5.3-5.5):

    .. math:: \hat g = \arg\min_{g \in G_K}
              \|\widehat E(Y|W) - \hat T g\|^2,

    with :math:`G_K` a K-dimensional sieve. Here K itself does the
    regularising: truncating the basis bounds the inverse, exactly as
    the Tikhonov penalty does in :mod:`morie.fn.hrztikr`. The two are
    alternative regularisations of the SAME ill-posed problem, and
    choosing K too large reproduces the instability the sieve was
    meant to prevent -- so the condition number at the chosen K is
    reported.

    Parameters
    ----------
    T : array-like, shape (m, k)
        Discretised operator.
    Ey_w : array-like, shape (m,)
        Estimated conditional mean.
    K : int, optional
        Sieve dimension; defaults to a conservative truncation.

    Returns
    -------
    RichResult
        keys: ``g`` (length k, zero-padded beyond K), ``K``,
        ``residual_norm``, ``condition_number_at_K``,
        ``regularisation`` ("truncation"), ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 5, Sec. 5.4.2 (estimation by series truncation when
    T is unknown).
    """
    Tm = np.atleast_2d(np.asarray(T, dtype=float))
    b = np.asarray(Ey_w, dtype=float).ravel()
    if Tm.shape[0] != b.size:
        raise ValueError(f"T has {Tm.shape[0]} rows but Ey_w has {b.size}.")
    k = Tm.shape[1]
    Kd = max(1, min(k, int(np.sqrt(Tm.shape[0])))) if K is None else int(K)
    if not 1 <= Kd <= k:
        raise ValueError(f"K must lie in 1..{k}, got {Kd}.")
    Tk = Tm[:, :Kd]
    gk, *_ = np.linalg.lstsq(Tk, b, rcond=None)
    g = np.zeros(k)
    g[:Kd] = gk
    cond = float(np.linalg.cond(Tk.T @ Tk))
    return RichResult(payload={"g": g, "K": Kd,
                               "residual_norm": float(np.linalg.norm(Tk @ gk - b)),
                               "condition_number_at_K": cond,
                               "regularisation": "truncation",
                               "method": "Sieve NPIV; K regularises exactly as alpha does"})


def cheatsheet():
    return "hrzsitr: K IS the regularisation -- too large and instability returns"


#: Catalogue alias for :func:`hrz_sieve_iv`.
horowitz_sieve_npiv = hrz_sieve_iv
