# morie.fn -- function file (rootcoder007/morie)
"""Bounded-Lipschitz metrisation of weak convergence."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_ch2_weak_convergence_lipschitz"]


def kosorok_ch2_weak_convergence_lipschitz(X_n, X, n_functions=200, rng=None):
    r"""Bounded-Lipschitz characterisation:

    .. math:: X_n \Rightarrow X \iff \sup_{f \in BL_1}
              |E^* f(X_n) - E f(X)| \to 0,

    where :math:`BL_1` is the class of functions bounded by 1 with
    Lipschitz constant at most 1. This metrises weak convergence, so
    the supremum IS a distance -- returned here as an estimate over a
    random sample of BL_1 members (cosine and clipped-linear forms),
    which lower-bounds the true supremum.

    Parameters
    ----------
    X_n, X : array-like
        Samples of the two laws (1-D).
    n_functions : int, default 200
        BL_1 members to sample.
    rng : numpy Generator, optional

    Returns
    -------
    RichResult
        keys: ``bl_distance`` (the max found), ``is_lower_bound``
        (True), ``n_functions``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (the bounded Lipschitz metric).
    """
    A = np.asarray(X_n, dtype=float).ravel()
    B = np.asarray(X, dtype=float).ravel()
    if A.size < 2 or B.size < 2:
        raise ValueError("both samples need at least 2 observations.")
    rng = np.random.default_rng(0) if rng is None else rng
    best = 0.0
    for _ in range(int(n_functions)):
        # both families lie in BL_1: |f| <= 1 and |f'| <= 1
        if rng.random() < 0.5:
            w = rng.uniform(0.1, 1.0)
            shift = rng.uniform(-3, 3)
            f = lambda z, w=w, s=shift: np.sin(w * (z - s)) / max(w, 1.0)
        else:
            s = rng.uniform(-3, 3)
            f = lambda z, s=s: np.clip(z - s, -1.0, 1.0)
        best = max(best, abs(float(np.mean(f(A))) - float(np.mean(f(B)))))
    return RichResult(
        payload={"bl_distance": float(best), "is_lower_bound": True,
                 "n_functions": int(n_functions),
                 "method": "sup over sampled BL_1 functions (lower bound on the metric)"}
    )


def cheatsheet():
    return "ksr039: BL_1 sup metrises weak convergence; sampled => lower bound"
