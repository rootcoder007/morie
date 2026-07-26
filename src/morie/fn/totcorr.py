# morie.fn -- function file (rootcoder007/morie)
"""Total correlation / multi-information (Watanabe 1960)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["total_correlation"]

_LOG2 = {"bits": np.log(2.0), "nats": 1.0}


def _entropy(p, base):
    """Shannon entropy of a (flattened) PMF, 0 log 0 = 0."""
    q = p[p > 0]
    return float(-(q * np.log(q)).sum() / _LOG2[base])


def _validate_pmf(p):
    arr = np.asarray(p, dtype=float)
    if arr.ndim < 2:
        raise ValueError(
            f"p must be the JOINT PMF over >= 2 variables, i.e. an array with one "
            f"axis per variable; got ndim={arr.ndim}. Total correlation of a single "
            "variable is identically 0 and carries no information."
        )
    if np.any(arr < 0):
        raise ValueError("p must be non-negative -- it is a probability mass function")
    total = float(arr.sum())
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"p must sum to 1; got {total!r}")
    return arr


def total_correlation(p, base="bits"):
    r"""Total correlation (multi-information) of a discrete joint distribution.

    .. math::

        TC(X_1,\ldots,X_n) = \sum_{i=1}^{n} H(X_i) - H(X_1,\ldots,X_n)

    Parameters
    ----------
    p : array-like
        Joint PMF with **one axis per variable**; must be non-negative and
        sum to 1. Shape ``(2, 2, 3)`` means three variables with 2, 2 and 3
        outcomes.
    base : {"bits", "nats"}, default ``"bits"``
        Logarithm base for the entropies.

    Returns
    -------
    RichResult
        keys: ``estimate`` (:math:`TC`), ``joint_entropy``,
        ``marginal_entropies``, ``n_vars``, ``base``, ``method``.

    Raises
    ------
    ValueError
        If ``p`` has fewer than 2 axes, is negative, or does not sum to 1.

    References
    ----------
    Watanabe, S. (1960). Information theoretical analysis of multivariate
        correlation. *IBM Journal of Research and Development*, 4(1), 66-82.

    Notes
    -----
    :math:`TC \ge 0`, with equality **iff** the variables are mutually
    independent -- it is the Kullback-Leibler divergence from the joint to
    the product of its marginals. For two variables it reduces exactly to the
    mutual information :math:`I(X_1; X_2)`.

    Total correlation is *not* dual total correlation
    (:mod:`morie.fn.dualtc`); the two coincide only for :math:`n = 2`.
    """
    if base not in _LOG2:
        raise ValueError(f"base must be 'bits' or 'nats'; got {base!r}")
    arr = _validate_pmf(p)
    n_vars = arr.ndim
    joint = _entropy(arr.ravel(), base)
    marginals = []
    for axis in range(n_vars):
        others = tuple(a for a in range(n_vars) if a != axis)
        marginals.append(_entropy(arr.sum(axis=others), base))
    tc = float(sum(marginals) - joint)
    return RichResult(
        payload={
            "estimate": tc,
            "joint_entropy": joint,
            "marginal_entropies": marginals,
            "n_vars": int(n_vars),
            "base": base,
            "method": "total correlation TC = sum H(Xi) - H(X1..Xn) (Watanabe 1960)",
        }
    )


def cheatsheet():
    return "totcorr: TC = sum_i H(X_i) - H(X_1..X_n) (Watanabe 1960)."
