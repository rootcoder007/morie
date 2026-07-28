# morie.fn -- function file (rootcoder007/morie)
"""Uniform covering number."""

import numpy as np

from ._kosorok import covering_number_grid
from ._richresult import RichResult

__all__ = ["kosorok_ch2_uniform_covering_number"]


def kosorok_ch2_uniform_covering_number(F, X, eps, r=2, n_measures=8, rng=None):
    r"""Uniform covering number

    .. math:: \sup_Q N\big(\epsilon \|F\|_{Q,r},\;
              \mathcal F,\; L_r(Q)\big),

    the supremum over ALL finitely discrete probability measures Q --
    which is what makes it *uniform* and hence usable without knowing
    P. Approximated here by maximising over ``n_measures`` random
    discrete measures supported on the sample; a supremum over an
    uncountable set cannot be computed exactly, and this returns a
    lower bound on it, stated as such.

    Note the radius scales with the envelope norm
    :math:`\|F\|_{Q,r}`, so the count is scale-free.

    Parameters
    ----------
    F : sequence of callables
        Class representatives.
    X : array-like
        Support points.
    eps : float in (0, 1)
        Relative radius.
    r : int, default 2
        L_r exponent.
    n_measures : int, default 8
        Random measures over which the sup is taken.
    rng : numpy Generator, optional

    Returns
    -------
    RichResult
        keys: ``covering_number`` (the max found), ``per_measure``,
        ``envelope_norms``, ``eps``, ``r``, ``is_lower_bound`` (True),
        ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (uniform entropy).
    """
    X = np.asarray(X, dtype=float).ravel()
    F = list(F)
    if not F:
        raise ValueError("F must contain at least one function.")
    eps = float(eps)
    if not 0 < eps < 1:
        raise ValueError(f"eps must lie in (0, 1), got {eps}.")
    r = int(r)
    if r < 1:
        raise ValueError(f"r must be at least 1, got {r}.")
    rng = np.random.default_rng(0) if rng is None else rng
    vals = np.array([f(X) for f in F], dtype=float)  # (|F|, n)
    env = np.max(np.abs(vals), axis=0)

    counts, norms = [], []
    for _ in range(int(n_measures)):
        w = rng.dirichlet(np.ones(X.size))
        env_norm = float((np.sum(w * np.abs(env) ** r)) ** (1.0 / r))
        if env_norm <= 0:
            continue
        # L_r(Q) distance between functions, weighted by this measure
        def dist(a, b, w=w, r=r):
            return float((np.sum(w * np.abs(a - b) ** r)) ** (1.0 / r))

        counts.append(covering_number_grid(vals, eps * env_norm, metric=dist))
        norms.append(env_norm)
    if not counts:
        raise ValueError("degenerate envelope; covering number undefined.")
    return RichResult(
        payload={"covering_number": int(max(counts)),
                 "per_measure": np.array(counts),
                 "envelope_norms": np.array(norms), "eps": eps, "r": r,
                 "is_lower_bound": True,
                 "method": "sup_Q N(eps ||F||_{Q,r}, F, L_r(Q)), sampled over Q"}
    )


def cheatsheet():
    return "ksr033: sup over Q is sampled, so the count is a LOWER bound"
