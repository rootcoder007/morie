# morie.fn -- function file (rootcoder007/morie)
"""Differentially private sum over a bounded range."""

from __future__ import annotations

import numpy as np

from ._dp import check_budget, clip_to_range
from ._richresult import RichResult

__all__ = ["dp_sum"]


def dp_sum(x, a, b, epsilon=1.0, seed=None):
    r"""Privately sum values clipped to ``[a, b]``.

    Clipping is not a convenience -- it is what makes the sensitivity finite
    at all. An unbounded value has unbounded influence, so no finite noise
    provides any guarantee. After clipping, replacing one record moves the sum
    by at most :math:`b - a`, giving Laplace scale :math:`(b-a)/\varepsilon`.

    The bounds must come from outside the data -- domain knowledge, a public
    schema, a prior release. Choosing them by looking at the sample (taking
    the observed min and max, say) is itself a non-private query and voids the
    guarantee. That is the most common way this function gets misused, so the
    clipped fraction is reported: a large one means the bounds are biting and
    the release is biased toward the interior.

    Parameters
    ----------
    x : array-like
        Values, one per record.
    a, b : float
        Clipping bounds with ``a < b``, chosen independently of the data.
    epsilon : float
        Privacy budget, positive.
    seed : int, optional
        Seed; leave ``None`` for a real release.

    Returns
    -------
    RichResult
        ``release``, ``true_sum``, ``noise_scale``, ``clipped_fraction``,
        ``sensitivity``.

    References
    ----------
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-407.

    Examples
    --------
    Sensitivity is the range, so wider bounds cost more noise.

    >>> narrow = dp_sum([1.0, 2.0], 0, 10, epsilon=1.0, seed=0)["noise_scale"]
    >>> wide = dp_sum([1.0, 2.0], 0, 100, epsilon=1.0, seed=0)["noise_scale"]
    >>> float(narrow), float(wide)
    (10.0, 100.0)

    Unbiased around the clipped truth.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> v = rng.uniform(0, 10, 200)
    >>> d = [dp_sum(v, 0, 10, 1.0, seed=s)["release"] for s in range(3000)]
    >>> bool(abs(float(np.mean(d)) - v.sum()) < 5.0)
    True

    Clipping is reported, so bounds that bite are visible.

    >>> r = dp_sum([1.0, 2.0, 50.0], 0, 10, epsilon=1.0, seed=0)
    >>> float(round(r["clipped_fraction"], 4)), float(r["true_sum"])
    (0.3333, 13.0)

    >>> dp_sum([1.0], 10, 0, epsilon=1.0)
    Traceback (most recent call last):
        ...
    ValueError: need a < b, got a=10.0, b=0.0
    """
    epsilon, _ = check_budget(epsilon)
    xc, a, b = clip_to_range(np.atleast_1d(np.asarray(x, dtype=float)).ravel(), a, b)
    raw = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    clipped_frac = float(np.mean(raw != xc)) if raw.size else 0.0
    sens = b - a
    scale = sens / epsilon
    rng = np.random.default_rng(seed)
    rel = float(xc.sum() + rng.laplace(0.0, scale))
    return RichResult(
        title="DP sum",
        summary_lines=[("epsilon", epsilon), ("bounds", f"[{a:g}, {b:g}]"),
                       ("noise scale", scale), ("clipped", clipped_frac)],
        warnings=(["more than 10% of values were clipped; the bounds are biting "
                   "and the release is biased toward the interior"]
                  if clipped_frac > 0.10 else []),
        payload={
            "release": rel, "true_sum": float(xc.sum()),
            "noise_scale": scale, "sensitivity": float(sens),
            "clipped_fraction": clipped_frac, "bounds": (a, b),
            "n": int(xc.size), "epsilon": epsilon, "method": "dp_sum",
        },
    )


def cheatsheet():
    return "dpsum: sensitivity = b - a, so bounds MUST come from outside the data or the guarantee is void"
