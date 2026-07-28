# morie.fn -- function file (rootcoder007/morie)
"""RLS cross-correlation vector."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_theta_vector"]


def rangayyan_ch3_rls_theta_vector(r, x, lam=0.99, n=None):
    r"""RLS exponentially weighted cross-correlation vector
    (Rangayyan Ch. 3):

    .. math:: \Theta(n) = \sum_{i=1}^{n} \lambda^{n-i}\,
              \mathbf{r}(i)\, x(i).

    Paired with :mod:`morie.fn.rng165`: the RLS weight vector solves
    :math:`\Phi(n)\mathbf{w}(n) = \Theta(n)`, which is the
    normal-equation form the recursion updates without ever inverting
    Phi directly. The solved weights are returned alongside.

    Parameters
    ----------
    r : array-like, shape (N, p)
        Reference vectors.
    x : array-like, shape (N,)
        Primary input.
    lam : float in (0, 1], default 0.99
        Forgetting factor.
    n : int, optional
        Time index.

    Returns
    -------
    RichResult
        keys: ``Theta``, ``weights`` (solving Phi w = Theta),
        ``lam``, ``n``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (the RLS algorithm).
    """
    from .rng165 import rangayyan_ch3_rls_phi_matrix

    R = np.atleast_2d(np.asarray(r, dtype=float))
    xv = np.asarray(x, dtype=float).ravel()
    if R.shape[0] != xv.size:
        R = R.T
    if R.shape[0] != xv.size:
        raise ValueError("r must have one row per sample of x.")
    lam = float(lam)
    if not 0 < lam <= 1:
        raise ValueError(f"lam must lie in (0, 1], got {lam}.")
    N = R.shape[0]
    idx = N if n is None else int(n)
    if not 1 <= idx <= N:
        raise ValueError(f"n must lie in 1..{N}, got {idx}.")
    w = lam ** (idx - 1 - np.arange(idx))
    Theta = (R[:idx] * w[:, None]).T @ xv[:idx]
    Phi = rangayyan_ch3_rls_phi_matrix(R, lam=lam, n=idx)["Phi"]
    try:
        weights = np.linalg.solve(Phi, Theta)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(Phi, Theta, rcond=None)[0]
    return RichResult(payload={"Theta": Theta, "weights": weights, "lam": lam,
                               "n": idx,
                               "method": "Theta(n) = sum lambda^(n-i) r(i) x(i); Phi w = Theta"})


def cheatsheet():
    return "rng166: RLS solves Phi w = Theta without inverting Phi"
