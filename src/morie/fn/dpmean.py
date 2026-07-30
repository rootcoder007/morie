# morie.fn -- function file (rootcoder007/morie)
"""Differentially private mean over a bounded range."""

from __future__ import annotations

import numpy as np

from ._dp import check_budget, clip_to_range
from ._richresult import RichResult

__all__ = ["dp_mean"]


def dp_mean(x, a, b, epsilon=1.0, seed=None, split=0.5, known_n=True):
    r"""Privately estimate a mean of values clipped to ``[a, b]``.

    With :math:`n` public, the mean has sensitivity :math:`(b-a)/n` and one
    Laplace draw suffices. When :math:`n` is itself private the budget must be
    **split** between a private sum and a private count, and the ratio of the
    two noisy quantities is no longer unbiased -- the expectation of a ratio
    is not the ratio of expectations.

    Naively releasing a private sum and dividing by the true :math:`n` leaks
    :math:`n`. This makes the choice explicit through ``known_n`` rather than
    letting it be an accident.

    Parameters
    ----------
    x : array-like
        Values, one per record.
    a, b : float
        Clipping bounds with ``a < b``, chosen independently of the data.
    epsilon : float
        Total privacy budget, positive.
    seed : int, optional
        Seed; leave ``None`` for a real release.
    split : float
        Fraction of the budget given to the sum when ``known_n=False``.
    known_n : bool
        Treat the sample size as public.

    Returns
    -------
    RichResult
        ``release``, ``true_mean``, ``sensitivity``, ``noise_scale``,
        ``clipped_fraction``, ``epsilon_sum``, ``epsilon_count``.

    References
    ----------
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-487.

    Examples
    --------
    With n public the sensitivity is the range over n, so noise shrinks with
    the sample.

    >>> small = dp_mean([1.0] * 10, 0, 1, epsilon=1.0, seed=0)["sensitivity"]
    >>> big = dp_mean([1.0] * 1000, 0, 1, epsilon=1.0, seed=0)["sensitivity"]
    >>> float(small), float(big)
    (0.1, 0.001)

    Unbiased when n is public.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> v = rng.uniform(0, 1, 500)
    >>> d = [dp_mean(v, 0, 1, 1.0, seed=s)["release"] for s in range(3000)]
    >>> bool(abs(float(np.mean(d)) - v.mean()) < 0.01)
    True

    A private n costs budget and is recorded as a split.

    >>> r = dp_mean(v, 0, 1, epsilon=1.0, seed=0, known_n=False)
    >>> float(r["epsilon_sum"]), float(r["epsilon_count"])
    (0.5, 0.5)

    >>> dp_mean([], 0, 1, epsilon=1.0)
    Traceback (most recent call last):
        ...
    ValueError: x must be non-empty
    """
    epsilon, _ = check_budget(epsilon)
    raw = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    if raw.size == 0:
        raise ValueError("x must be non-empty")
    xc, a, b = clip_to_range(raw, a, b)
    n = xc.size
    clipped_frac = float(np.mean(raw != xc))
    rng = np.random.default_rng(seed)
    if known_n:
        sens = (b - a) / n
        scale = sens / epsilon
        rel = float(xc.mean() + rng.laplace(0.0, scale))
        eps_s, eps_c = epsilon, 0.0
    else:
        if not 0.0 < split < 1.0:
            raise ValueError("split must be in (0, 1)")
        eps_s, eps_c = epsilon * split, epsilon * (1.0 - split)
        noisy_sum = xc.sum() + rng.laplace(0.0, (b - a) / eps_s)
        noisy_n = max(n + rng.laplace(0.0, 1.0 / eps_c), 1.0)
        rel = float(noisy_sum / noisy_n)
        sens = (b - a) / n
        scale = (b - a) / eps_s / n
    return RichResult(
        title="DP mean",
        summary_lines=[("epsilon", epsilon), ("n", int(n)),
                       ("sensitivity", float(sens)), ("release", rel)],
        warnings=(["n was treated as private, so the release is a ratio of two "
                   "noisy quantities and is not unbiased"] if not known_n else []),
        payload={
            "release": rel, "true_mean": float(xc.mean()),
            "sensitivity": float(sens), "noise_scale": float(scale),
            "clipped_fraction": clipped_frac, "bounds": (a, b), "n": int(n),
            "epsilon": epsilon, "epsilon_sum": float(eps_s),
            "epsilon_count": float(eps_c), "known_n": bool(known_n),
            "method": "dp_mean",
        },
    )


def cheatsheet():
    return "dpmean: sensitivity (b-a)/n with n public; a PRIVATE n needs a budget split and is biased"
