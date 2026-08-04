# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stochastic gradient boosting subsampling (ESL Ch 10.12.2)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_subsampling"]


def _lcg_uniforms(count, seed=13):
    """Shared exact-integer LCG, matching the rest of the package."""
    s = int(seed)
    out = np.empty(count)
    for i in range(count):
        s = (1664525 * s + 1013904223) % 2 ** 32
        out[i] = (s + 0.5) / 2 ** 32
    return out


def esl_subsampling(eta, n=None, seed=13):
    """
    Stochastic gradient boosting: use an eta-fraction sample at each
    stage.

    ESL Ch 10.12.2 (following Friedman) draws a fraction eta of the
    training data WITHOUT replacement at every boosting iteration.
    Two effects, and the second is the one usually forgotten: it
    reduces variance by decorrelating the stages, and it cuts the
    cost per stage by the same factor, so a run with eta = 0.5 is
    both better regularised and twice as fast. Friedman's suggested
    default is eta = 0.5, smaller for large n.

    With ``n`` supplied this actually draws the subsample, using the
    shared exact-integer LCG so the selection is reproducible across
    languages rather than depending on a global RNG.

    Parameters
    ----------
    eta : float
        Sampling fraction in (0, 1].
    n : int, optional
        Training set size; if given, an index subsample is returned.
    seed : int
        LCG seed.

    Returns
    -------
    result : dict
        Keys: estimate (subsample size, or eta when n is absent),
        eta, n, n_sampled, indices (0-based, sorted),
        cost_multiplier, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 10.12.2;
    Friedman (2002).

    Examples
    --------
    >>> out = esl_subsampling(0.5, n=10)
    >>> out["n_sampled"]
    5
    >>> len(set(out["indices"])) == 5      # sampled without replacement
    True
    >>> all(0 <= i < 10 for i in out["indices"])
    True
    >>> out["cost_multiplier"]
    0.5

    The draw is deterministic for a given seed:

    >>> esl_subsampling(0.5, n=10)["indices"] == esl_subsampling(0.5, n=10)["indices"]
    True
    >>> esl_subsampling(0.5, n=10, seed=7)["indices"] == out["indices"]
    False
    >>> esl_subsampling(0.0, n=10)
    Traceback (most recent call last):
        ...
    ValueError: the sampling fraction must lie in (0, 1]; got 0.0.
    """
    eta = float(eta)
    if not 0 < eta <= 1:
        raise ValueError(f"the sampling fraction must lie in (0, 1]; got {eta}.")
    idx = None
    n_s = None
    if n is not None:
        n = int(n)
        if n < 1:
            raise ValueError(f"n must be >= 1; got {n}.")
        n_s = max(1, int(round(eta * n)))
        u = _lcg_uniforms(n, seed)
        idx = sorted(int(v) for v in np.argsort(u, kind="stable")[:n_s])
    return RichResult(payload={
        "estimate": n_s if n_s is not None else eta,
        "eta": eta, "n": n, "n_sampled": n_s, "indices": idx,
        "cost_multiplier": eta,
        "method": "stochastic gradient boosting subsample, LCG-deterministic, no replacement"})


def cheatsheet():
    return "eslsmp: eta-fraction per stage; regularises AND cuts cost by eta; LCG draw"


# compact alias per ledger/NAMING.md
eslsubsampling = esl_subsampling
