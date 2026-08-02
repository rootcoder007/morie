# morie.fn -- function file (rootcoder007/morie)
"""Dual total correlation (Han 1978)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from .totcorr import _entropy, _validate_pmf

__all__ = ["dual_total_correlation"]


def dual_total_correlation(p, base="bits"):
    r"""Dual total correlation (excess entropy / binding information).

    .. math::

        DTC(X_1,\ldots,X_n) = H(X_1,\ldots,X_n)
            - \sum_{i=1}^{n} H(X_i \mid X_{-i})

    where :math:`X_{-i}` is every variable except :math:`X_i`, and each
    conditional entropy is evaluated as
    :math:`H(X_i \mid X_{-i}) = H(X_1,\ldots,X_n) - H(X_{-i})`.

    Parameters
    ----------
    p : array-like
        Joint PMF with one axis per variable; non-negative, summing to 1.
    base : {"bits", "nats"}, default ``"bits"``

    Returns
    -------
    RichResult
        keys: ``estimate`` (:math:`DTC`), ``joint_entropy``,
        ``conditional_entropies``, ``n_vars``, ``base``, ``method``.

    Raises
    ------
    ValueError
        If ``p`` has fewer than 2 axes, is negative, or does not sum to 1.

    References
    ----------
    Han, T. S. (1978). Nonnegative entropy measures of multivariate symmetric
        correlations. *Information and Control*, 36(2), 133-156.

    Notes
    -----
    :math:`DTC \ge 0`, and it vanishes iff the variables are mutually
    independent -- the same zero set as total correlation, but a different
    quantity. They agree only at :math:`n = 2`, where both equal the mutual
    information; for :math:`n \ge 3` they measure different things and
    :math:`DTC` may be larger or smaller than :math:`TC`.

    The conditional entropies are obtained by the chain rule from marginals
    of the joint rather than by building :math:`n` conditional tables, which
    avoids dividing by zero-probability conditioning events.
    """
    if base not in ("bits", "nats"):
        raise ValueError(f"base must be 'bits' or 'nats'; got {base!r}")
    arr = _validate_pmf(p)
    n_vars = arr.ndim
    joint = _entropy(arr.ravel(), base)
    conditionals = []
    for axis in range(n_vars):
        # H(X_i | X_-i) = H(X_1..X_n) - H(X_-i)
        h_rest = _entropy(arr.sum(axis=axis).ravel(), base)
        conditionals.append(joint - h_rest)
    dtc = float(joint - sum(conditionals))
    return RichResult(
        payload={
            "estimate": dtc,
            "joint_entropy": joint,
            "conditional_entropies": conditionals,
            "n_vars": int(n_vars),
            "base": base,
            "method": "dual total correlation DTC = H(X) - sum H(Xi | X_-i) (Han 1978)",
        }
    )


def cheatsheet():
    return "dualtc: DTC = H(X_1..X_n) - sum_i H(X_i | X_-i) (Han 1978)."
