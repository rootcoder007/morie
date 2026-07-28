# morie.fn -- function file (rootcoder007/morie)
"""Bracketing Glivenko-Cantelli theorem."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_ch2_glivenko_cantelli_bracketing"]


def kosorok_ch2_glivenko_cantelli_bracketing(F, X, eps_grid=None, P=None):
    r"""Bracketing Glivenko-Cantelli theorem:

    if :math:`N_{[\,]}(\epsilon, \mathcal F, L_1(P)) < \infty` for
    every :math:`\epsilon > 0`, then :math:`\mathcal F` is
    P-Glivenko-Cantelli.

    Computes the bracketing numbers at a grid of eps by building
    brackets from the pointwise minimum and maximum of nearby
    functions, and reports whether they are finite throughout -- the
    hypothesis of the theorem. Finiteness at every eps in a finite
    grid does NOT prove finiteness at every eps > 0; the flag is
    named ``finite_on_grid`` for that reason.

    Parameters
    ----------
    F : sequence of callables
        Class representatives.
    X : array-like
        Sample (the L_1(P) norm is taken empirically).
    eps_grid : sequence of float, optional
        Radii at which to count.
    P : ignored
        Interface compatibility; the empirical measure is used.

    Returns
    -------
    RichResult
        keys: ``eps_grid``, ``bracketing_numbers``,
        ``finite_on_grid``, ``monotone`` (counts non-increasing in
        eps), ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (bracketing entropy and the GC theorem).
    """
    X = np.asarray(X, dtype=float).ravel()
    F = list(F)
    if not F:
        raise ValueError("F must contain at least one function.")
    vals = np.array([f(X) for f in F], dtype=float)
    if eps_grid is None:
        eps_grid = [0.5, 0.2, 0.1, 0.05]
    eps_grid = [float(e) for e in eps_grid]
    if any(e <= 0 for e in eps_grid):
        raise ValueError("eps values must be positive.")

    counts = []
    for eps in eps_grid:
        # greedy bracketing: a bracket [l, u] covers f if l <= f <= u
        # pointwise and P(u - l) <= eps
        remaining = list(range(vals.shape[0]))
        n_br = 0
        while remaining:
            seed = remaining[0]
            lo = vals[seed].copy()
            hi = vals[seed].copy()
            covered = [seed]
            for j in remaining[1:]:
                nlo = np.minimum(lo, vals[j])
                nhi = np.maximum(hi, vals[j])
                if float(np.mean(nhi - nlo)) <= eps:
                    lo, hi = nlo, nhi
                    covered.append(j)
            remaining = [j for j in remaining if j not in covered]
            n_br += 1
        counts.append(n_br)
    counts = np.array(counts)
    return RichResult(
        payload={"eps_grid": np.array(eps_grid), "bracketing_numbers": counts,
                 "finite_on_grid": bool(np.all(np.isfinite(counts))),
                 "monotone": bool(np.all(np.diff(counts) >= 0)),
                 "method": "N_[](eps, F, L1(P_n)) by greedy bracketing (Kosorok Ch. 2)"}
    )


def cheatsheet():
    return "ksr034: finite bracketing numbers => GC; flag says 'on grid', not 'all eps'"
