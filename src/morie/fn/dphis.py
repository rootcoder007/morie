# morie.fn -- function file (rootcoder007/morie)
"""Differentially private histogram."""

from __future__ import annotations

import numpy as np

from ._dp import check_budget
from ._richresult import RichResult

__all__ = ["dp_histogram"]


def dp_histogram(x, bins=10, epsilon=1.0, range_=None, seed=None, nonneg=True):
    r"""Release a histogram privately.

    A histogram over **disjoint** bins is a rare bargain: moving one record
    changes two counts by one each, so under bounded DP the L1 sensitivity is
    2 and the *same* :math:`\varepsilon` covers every bin at once. Parallel
    composition, not sequential -- the budget is not divided by the number of
    bins, and treating it as though it were wastes accuracy for nothing.

    Bin edges must be fixed independently of the data, for the same reason
    clipping bounds must be. Deriving them from the observed range is a
    non-private query.

    Counts may go negative; clamping is post-processing and free, but biases
    small bins upward, so ``raw`` is kept.

    Parameters
    ----------
    x : array-like
        Values.
    bins : int or array-like
        Bin count or explicit edges.
    epsilon : float
        Privacy budget, positive -- covers the whole histogram.
    range_ : tuple, optional
        ``(low, high)`` for the binning, chosen independently of the data.
    seed : int, optional
        Seed; leave ``None`` for a real release.
    nonneg : bool
        Clamp released counts at zero.

    Returns
    -------
    RichResult
        ``release``, ``raw``, ``true_counts``, ``edges``, ``noise_scale``.

    References
    ----------
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-487.

    Examples
    --------
    One epsilon covers every bin -- the noise scale does not grow with the
    number of bins.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> v = rng.normal(0, 1, 5000)
    >>> a = dp_histogram(v, bins=5, epsilon=1.0, range_=(-4, 4), seed=0)
    >>> b = dp_histogram(v, bins=50, epsilon=1.0, range_=(-4, 4), seed=0)
    >>> float(a["noise_scale"]), float(b["noise_scale"])
    (2.0, 2.0)

    Total count is approximately preserved and each bin is unbiased.

    >>> bool(abs(a["release"].sum() - 5000) < 60)
    True

    >>> dp_histogram([1.0], bins=0, epsilon=1.0)
    Traceback (most recent call last):
        ...
    ValueError: bins must be a positive integer or an array of edges
    """
    epsilon, _ = check_budget(epsilon)
    v = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    if np.isscalar(bins) or (isinstance(bins, (int, np.integer))):
        if int(bins) < 1:
            raise ValueError("bins must be a positive integer or an array of edges")
    counts, edges = np.histogram(v, bins=bins, range=range_)
    # Disjoint bins -> parallel composition: sensitivity 2 for the whole vector.
    sens = 2.0
    scale = sens / epsilon
    rng = np.random.default_rng(seed)
    raw = counts.astype(float) + rng.laplace(0.0, scale, counts.size)
    rel = np.maximum(raw, 0.0) if nonneg else raw
    return RichResult(
        title="DP histogram",
        summary_lines=[("epsilon", epsilon), ("bins", int(counts.size)),
                       ("noise scale", scale)],
        payload={
            "release": rel, "raw": raw, "true_counts": counts,
            "edges": edges, "noise_scale": scale, "sensitivity": sens,
            "epsilon": epsilon, "method": "dp_histogram",
        },
    )


def cheatsheet():
    return "dphis: disjoint bins = PARALLEL composition; one epsilon covers all bins, do not divide it"
