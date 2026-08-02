# morie.fn -- function file (rootcoder007/morie)
"""Differentially private min and max."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from .dpqua import dp_quantile

__all__ = ["dp_minmax"]


def dp_minmax(x, epsilon=1.0, a=None, b=None, alpha=0.01, seed=None):
    r"""Release approximate extremes privately.

    The true minimum and maximum are the *least* private statistics there are:
    each is determined by a single record, so releasing either with any useful
    accuracy leaks that record almost completely. There is no honest private
    version of "the maximum".

    What is releasable is an inner quantile pair -- the :math:`\alpha` and
    :math:`1-\alpha` quantiles -- each obtained by the rank-based exponential
    mechanism, with the budget split between them. The result is a robust
    *range estimate*, not the extremes, and it is labelled as such.

    A common use is deriving clipping bounds for :func:`~morie.fn.dpsum.dp_sum`
    or :func:`~morie.fn.dpmean.dp_mean` without spending a non-private look at
    the data; the budget used here must be counted against the total.

    Parameters
    ----------
    x : array-like
        Values.
    epsilon : float
        Total budget, split evenly between the two quantiles.
    a, b : float, optional
        Clipping bounds chosen independently of the data.
    alpha : float
        Tail fraction, in (0, 0.5).
    seed : int, optional
        Seed; leave ``None`` for a real release.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``alpha``, ``epsilon_each``, ``true_min``,
        ``true_max``.

    References
    ----------
    Smith, A. (2011). Privacy-preserving statistical estimation with optimal
        convergence rates. *STOC 2011*, 813-822.

    Examples
    --------
    The released interval sits inside the true range, because it is a quantile
    pair rather than the extremes.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> v = rng.uniform(0, 100, 5000)
    >>> r = dp_minmax(v, epsilon=2.0, a=0, b=100, alpha=0.01, seed=1)
    >>> bool(r["lower"] > r["true_min"] - 5 and r["upper"] < r["true_max"] + 5)
    True
    >>> bool(r["lower"] < r["upper"])
    True

    The budget is split, and that is reported rather than assumed.

    >>> float(r["epsilon_each"])
    1.0

    >>> dp_minmax([1.0, 2.0], epsilon=1.0, alpha=0.9)
    Traceback (most recent call last):
        ...
    ValueError: alpha must be in (0, 0.5)
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    v = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    half = float(epsilon) / 2.0
    lo = dp_quantile(v, q=alpha, epsilon=half, a=a, b=b, seed=seed)
    hi = dp_quantile(v, q=1.0 - alpha, epsilon=half, a=a, b=b,
                     seed=None if seed is None else seed + 1)
    warn = list(lo.warnings)
    warn.append(
        "these are the alpha and 1-alpha quantiles, NOT the minimum and "
        "maximum -- the true extremes are each determined by one record and "
        "cannot be released privately"
    )
    return RichResult(
        title="DP range (inner quantiles)",
        summary_lines=[("epsilon", float(epsilon)), ("alpha", float(alpha)),
                       ("lower", lo["release"]), ("upper", hi["release"])],
        warnings=warn,
        payload={
            "lower": lo["release"], "upper": hi["release"],
            "alpha": float(alpha), "epsilon": float(epsilon),
            "epsilon_each": half,
            "true_min": float(v.min()), "true_max": float(v.max()),
            "method": "dp_minmax",
        },
    )


def cheatsheet():
    return "dpmnp: there is NO private max -- returns alpha/1-alpha quantiles and says so"
