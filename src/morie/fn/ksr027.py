# morie.fn -- function file (rootcoder007/morie)
"""Pointwise strong law for the EDF."""

import numpy as np

from ._kosorok import empirical_df
from ._richresult import RichResult

__all__ = ["kosorok_ch2_law_large_numbers_pointwise"]


def kosorok_ch2_law_large_numbers_pointwise(X, t, F=None, n_grid=None):
    r"""Pointwise strong law: :math:`F_n(t) \to F(t)` almost surely
    for each fixed t.

    A limit statement has no single-sample value, so this returns the
    witness: :math:`|F_n(t) - F(t)|` along an increasing sequence of
    sample sizes drawn from the given sample, which must shrink. Note
    this is POINTWISE -- the uniform version is Glivenko-Cantelli
    (:mod:`morie.fn.ksr028`) and does not follow from this one without
    an argument.

    Parameters
    ----------
    X : array-like
        Sample.
    t : float
        The fixed point.
    F : callable, optional
        True CDF; the uniform CDF on [0, 1] if omitted.
    n_grid : sequence of int, optional
        Sample sizes at which to evaluate; geometric by default.

    Returns
    -------
    RichResult
        keys: ``n_grid``, ``deviation``, ``F_t``, ``shrinking``
        (bool), ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (classical empirical process results).
    """
    X = np.asarray(X, dtype=float).ravel()
    N = X.size
    if N < 8:
        raise ValueError(f"need at least 8 observations, got {N}.")
    t = float(t)
    Ft = float(np.clip(t, 0, 1)) if F is None else float(F(t))
    if n_grid is None:
        n_grid = [max(4, int(N * f)) for f in (0.1, 0.25, 0.5, 1.0)]
    n_grid = [int(g) for g in n_grid]
    if any(g < 1 or g > N for g in n_grid):
        raise ValueError(f"n_grid entries must lie in 1..{N}.")
    dev = np.array([abs(float(empirical_df(X[:g], t)[0]) - Ft) for g in n_grid])
    return RichResult(
        payload={"n_grid": np.array(n_grid), "deviation": dev, "F_t": Ft,
                 "shrinking": bool(dev[-1] <= dev[0] + 1e-12),
                 "method": "Pointwise |F_n(t) - F(t)| witness (Kosorok Ch. 2)"}
    )


def cheatsheet():
    return "ksr027: pointwise SLLN witness; NOT the uniform statement"
