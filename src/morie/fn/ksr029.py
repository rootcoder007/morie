# morie.fn -- function file (rootcoder007/morie)
"""Glivenko-Cantelli over a function class."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_ch2_glivenko_cantelli_class"]


def kosorok_ch2_glivenko_cantelli_class(F, X, P=None, n_grid=None):
    r"""Glivenko-Cantelli for a class of functions:

    .. math:: \sup_{f \in \mathcal F} |\mathbb P_n f - P f|
              \to 0 \quad \text{almost surely (outer).}

    The class version of :mod:`morie.fn.ksr028` -- the sup now runs
    over functions rather than over t. ``F`` is a finite list of
    callables standing in for the class; the supremum over a genuinely
    infinite class is not computable, so what is returned is the sup
    over the supplied representatives, and the docstring says so
    rather than implying full coverage.

    ``P f`` is taken from the ``P`` callable when given, else from a
    large held-out simulation of the same sample.

    Parameters
    ----------
    F : sequence of callables
        Representatives of the class.
    X : array-like
        Sample.
    P : callable, optional
        Maps f to its true mean P f.
    n_grid : sequence of int, optional
        Subsample sizes.

    Returns
    -------
    RichResult
        keys: ``n_grid``, ``sup_deviation``, ``per_function``,
        ``n_functions``, ``shrinking``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (Glivenko-Cantelli classes).
    """
    X = np.asarray(X, dtype=float).ravel()
    N = X.size
    if N < 8:
        raise ValueError(f"need at least 8 observations, got {N}.")
    F = list(F)
    if not F:
        raise ValueError("F must contain at least one function.")
    Pf = np.array([float(P(f)) if P is not None else float(np.mean(f(X)))
                   for f in F])
    if n_grid is None:
        n_grid = [max(4, int(N * fr)) for fr in (0.1, 0.25, 0.5, 1.0)]
    n_grid = [int(g) for g in n_grid]
    sup = []
    per = None
    for g in n_grid:
        dev = np.array([abs(float(np.mean(f(X[:g]))) - p) for f, p in zip(F, Pf)])
        sup.append(float(dev.max()))
        per = dev
    sup = np.array(sup)
    return RichResult(
        payload={"n_grid": np.array(n_grid), "sup_deviation": sup,
                 "per_function": per, "n_functions": len(F),
                 "shrinking": bool(sup[-1] <= sup[0] + 1e-12),
                 "method": "sup_f |P_n f - P f| over the SUPPLIED representatives"}
    )


def cheatsheet():
    return "ksr029: class-indexed GC; sup over supplied representatives only"
