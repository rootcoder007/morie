# morie.fn -- function file (rootcoder007/morie)
"""NPIV operator estimate."""

import numpy as np

from ._richresult import RichResult

__all__ = ["hrz_npiv_operator"]


def hrz_npiv_operator(X, W, K=5, kind="poly"):
    r"""Kernel/sieve estimate of the NPIV operator (Horowitz Ch. 6):

    .. math:: (Tg)(w) = \int g(x)\, f_{X|W}(x|w)\, dx,

    represented on a sieve basis as :math:`\hat T_{jk} =
    \widehat E[p_k(X)\,q_j(W)]`. The operator's singular values
    decay to zero -- that decay IS the ill-posedness, and how fast it
    decays determines whether the problem is mildly or severely
    ill-posed. The returned singular values make that visible instead
    of leaving it as an abstraction.

    Parameters
    ----------
    X : array-like, shape (n,)
        Endogenous regressor.
    W : array-like, shape (n,)
        Instrument.
    K : int, default 5
        Sieve dimension for both bases.
    kind : {"poly", "fourier"}
        Basis type.

    Returns
    -------
    RichResult
        keys: ``T``, ``singular_values``, ``decay_ratio``
        (smallest/largest), ``severity`` ("mild"/"severe"), ``K``,
        ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 6 (the NPIV operator and ill-posedness).
    """
    from ._horowitz import sieve_basis

    X = np.asarray(X, dtype=float).ravel()
    W = np.asarray(W, dtype=float).ravel()
    if X.size != W.size:
        raise ValueError("X and W must have the same length.")
    K = int(K)
    if K < 1 or K > X.size:
        raise ValueError(f"K must lie in 1..{X.size}, got {K}.")
    P = sieve_basis(X, K=K, kind=kind)
    Q = sieve_basis(W, K=K, kind=kind)
    T = Q.T @ P / X.size
    sv = np.linalg.svd(T, compute_uv=False)
    ratio = float(sv[-1] / sv[0]) if sv[0] > 0 else 0.0
    return RichResult(payload={"T": T, "singular_values": sv,
                               "decay_ratio": ratio,
                               "severity": "severe" if ratio < 1e-6 else "mild",
                               "K": K, "n": int(X.size),
                               "method": "T_jk = E[p_k(X) q_j(W)]; singular decay IS the ill-posedness"})


def cheatsheet():
    return "hrznpivt: singular values decaying to zero is what makes NPIV ill-posed"
