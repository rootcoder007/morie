# morie.fn -- function file (rootcoder007/morie)
"""Differentially private quantile via the exponential mechanism."""

from __future__ import annotations

import numpy as np

from ._dp import check_budget, clip_to_range
from ._richresult import RichResult

__all__ = ["dp_quantile"]


def dp_quantile(x, q=0.5, epsilon=1.0, a=None, b=None, seed=None):
    r"""Release a quantile privately by selecting an interval, not by adding noise.

    A quantile has enormous worst-case sensitivity -- moving one record can
    shift the median across the whole data range -- so the Laplace mechanism is
    the wrong tool. Instead the sorted gaps between clipped values are treated
    as candidate intervals and one is selected by the exponential mechanism with
    utility

    .. math::
        u(i) = -\left| i - q n \right|,

    the distance from the target rank. A uniform draw within the selected
    interval then gives the release. Because utility depends only on *ranks*,
    the sensitivity is 1 however extreme the values are, which is exactly what
    makes this work where noise addition fails.

    Parameters
    ----------
    x : array-like
        Values.
    q : float
        Quantile in (0, 1).
    epsilon : float
        Privacy budget, positive.
    a, b : float, optional
        Clipping bounds, chosen independently of the data. Default to the
        observed range with a warning, which is convenient and *not private*.
    seed : int, optional
        Seed; leave ``None`` for a real release.

    Returns
    -------
    RichResult
        ``release``, ``true_quantile``, ``interval``, ``epsilon``.

    References
    ----------
    Smith, A. (2011). Privacy-preserving statistical estimation with optimal
        convergence rates. *STOC 2011*, 813-822.
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-487.

    Examples
    --------
    Rank-based utility means one wild value cannot drag the release with it --
    the failure mode that rules out adding noise to a quantile.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> v = rng.normal(0, 1, 2000)
    >>> med = dp_quantile(v, 0.5, epsilon=1.0, a=-5, b=5, seed=1)["release"]
    >>> bool(abs(med) < 0.5)
    True
    >>> wild = np.r_[v, 1e6]
    >>> med2 = dp_quantile(wild, 0.5, epsilon=1.0, a=-5, b=5, seed=1)["release"]
    >>> bool(abs(med2) < 0.5)
    True

    Larger epsilon concentrates the release near the true quantile.

    >>> errs = [abs(dp_quantile(v, 0.5, epsilon=e, a=-5, b=5, seed=2)["release"])
    ...         for e in (0.05, 20.0)]
    >>> bool(errs[1] < errs[0])
    True

    Omitting bounds is allowed but flagged, since taking them from the data is
    itself a private query.

    >>> bool(dp_quantile(v, 0.5, epsilon=1.0, seed=1).warnings)
    True

    >>> dp_quantile([1.0, 2.0], q=1.5, epsilon=1.0)
    Traceback (most recent call last):
        ...
    ValueError: q must be in (0, 1)
    """
    epsilon, _ = check_budget(epsilon)
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    v = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    if v.size == 0:
        raise ValueError("x must be non-empty")
    warn = []
    if a is None or b is None:
        a, b = float(v.min()), float(v.max())
        if a == b:
            a, b = a - 0.5, b + 0.5
        warn.append(
            "bounds were taken from the data, which is itself a non-private "
            "query; supply `a` and `b` from outside the data for a real release"
        )
    vc, a, b = clip_to_range(v, a, b)
    s = np.sort(vc)
    edges = np.r_[a, s, b]
    n = s.size
    gaps = np.diff(edges)
    # Utility depends only on rank, so sensitivity is 1 whatever the values are.
    ranks = np.arange(n + 1)
    util = -np.abs(ranks - q * n)
    logp = epsilon * util / 2.0 + np.log(np.maximum(gaps, 1e-300))
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    rng = np.random.default_rng(seed)
    i = int(rng.choice(n + 1, p=p))
    rel = float(rng.uniform(edges[i], edges[i + 1]))
    return RichResult(
        title=f"DP quantile (q={q:g})",
        summary_lines=[("epsilon", epsilon), ("n", int(n)), ("release", rel)],
        warnings=warn,
        payload={
            "release": rel, "true_quantile": float(np.quantile(vc, q)),
            "interval": (float(edges[i]), float(edges[i + 1])),
            "probabilities": p, "bounds": (a, b), "q": float(q),
            "n": int(n), "epsilon": epsilon, "method": "dp_quantile",
        },
    )


def cheatsheet():
    return "dpqua: exponential mechanism on RANK gaps -- sensitivity 1 however extreme the values"
