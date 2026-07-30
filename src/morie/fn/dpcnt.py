# morie.fn -- function file (rootcoder007/morie)
"""Differentially private count."""

from __future__ import annotations

import numpy as np

from ._dp import check_budget
from ._richresult import RichResult

__all__ = ["dp_count"]


def dp_count(D, epsilon=1.0, predicate=None, seed=None, nonneg=True):
    r"""Privately count records, optionally those satisfying a predicate.

    A count has L1 sensitivity **1** under bounded DP (one record replaced),
    so the Laplace scale is :math:`1/\varepsilon` regardless of how many
    records there are. That the noise does not grow with :math:`n` is what
    makes counts the cheapest thing to release privately -- relative error
    falls like :math:`1/n`.

    Under *unbounded* DP (a record added or removed) the sensitivity is still
    1 for a plain count, but 2 for a difference of counts; if you are
    releasing both a count and its complement, budget accordingly.

    The noisy count can be negative, which is impossible for a count. Clamping
    it at zero (``nonneg=True``, the default) is post-processing and costs no
    privacy, but it does introduce a small upward bias -- ``raw`` keeps the
    unclamped value so the bias is inspectable.

    Parameters
    ----------
    D : array-like
        Records, or a boolean/0-1 indicator vector.
    epsilon : float
        Privacy budget, positive.
    predicate : callable, optional
        ``predicate(record) -> bool``. Without it, ``D`` is taken as
        indicators when it is 0/1 and otherwise its length is counted.
    seed : int, optional
        Seed; leave ``None`` for a real release.
    nonneg : bool
        Clamp the release at zero.

    Returns
    -------
    RichResult
        ``release``, ``raw``, ``true_count``, ``noise_scale``, ``epsilon``.

    References
    ----------
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-487.

    Examples
    --------
    Noise scale is 1/epsilon and does not depend on n.

    >>> import numpy as np
    >>> a = dp_count(np.ones(10), epsilon=0.5, seed=0)["noise_scale"]
    >>> b = dp_count(np.ones(10_000), epsilon=0.5, seed=0)["noise_scale"]
    >>> float(a), float(b)
    (2.0, 2.0)

    Unbiased before clamping.

    >>> d = [dp_count(np.ones(100), 1.0, seed=s, nonneg=False)["release"]
    ...      for s in range(4000)]
    >>> bool(abs(float(np.mean(d)) - 100.0) < 0.2)
    True

    A predicate counts the matching records.

    >>> int(dp_count([1, 5, 9, 12], epsilon=10.0, predicate=lambda r: r > 4,
    ...              seed=3)["true_count"])
    3

    >>> dp_count([1, 2], epsilon=-1.0)
    Traceback (most recent call last):
        ...
    ValueError: epsilon must be finite and positive
    """
    epsilon, _ = check_budget(epsilon)
    arr = np.asarray(D)
    if predicate is not None:
        true_count = float(sum(bool(predicate(r)) for r in arr))
    elif arr.dtype == bool:
        true_count = float(arr.sum())
    elif arr.ndim == 1 and np.all(np.isin(arr, (0, 1))):
        true_count = float(arr.sum())
    else:
        true_count = float(len(arr))
    rng = np.random.default_rng(seed)
    b = 1.0 / epsilon
    raw = true_count + float(rng.laplace(0.0, b))
    rel = max(raw, 0.0) if nonneg else raw
    return RichResult(
        title="DP count",
        summary_lines=[("epsilon", epsilon), ("noise scale", b),
                       ("release", rel)],
        payload={
            "release": rel, "raw": raw, "true_count": true_count,
            "noise_scale": b, "sensitivity": 1.0, "epsilon": epsilon,
            "clamped": bool(nonneg and raw < 0),
            "method": "dp_count",
        },
    )


def cheatsheet():
    return "dpcnt: sensitivity 1 whatever n, so noise is O(1) and relative error falls like 1/n"
