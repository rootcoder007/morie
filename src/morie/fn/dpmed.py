# morie.fn -- function file (rootcoder007/morie)
"""Differentially private median."""

from __future__ import annotations

from ._richresult import RichResult
from .dpqua import dp_quantile

__all__ = ["dp_median"]


def dp_median(x, epsilon=1.0, a=None, b=None, seed=None):
    r"""Release the median privately.

    The median at :math:`q = 0.5` of
    :func:`~morie.fn.dpqua.dp_quantile`, selecting an inter-order-statistic
    gap by the exponential mechanism rather than adding noise.

    The median is the standard illustration of why noise addition fails for
    order statistics: its *global* sensitivity is the width of the data domain
    -- one record can move it from one end to the other in the worst case --
    even though its *local* sensitivity on any real dataset is tiny. Selecting
    on rank sidesteps the gap between the two.

    Parameters
    ----------
    x : array-like
        Values.
    epsilon : float
        Privacy budget, positive.
    a, b : float, optional
        Clipping bounds chosen independently of the data.
    seed : int, optional
        Seed; leave ``None`` for a real release.

    Returns
    -------
    RichResult
        ``release``, ``true_median``, ``interval``, ``epsilon``.

    References
    ----------
    Nissim, K., Raskhodnikova, S., & Smith, A. (2007). Smooth sensitivity and
        sampling in private data analysis. *STOC 2007*, 75-84.
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-487.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> v = rng.normal(5.0, 1.0, 3000)
    >>> r = dp_median(v, epsilon=1.0, a=0, b=10, seed=1)
    >>> bool(abs(r["release"] - 5.0) < 0.5)
    True

    Robust to a single extreme value, unlike a noisy mean.

    >>> w = np.r_[v, 1e9]
    >>> bool(abs(dp_median(w, epsilon=1.0, a=0, b=10, seed=1)["release"] - 5.0) < 0.5)
    True
    """
    r = dp_quantile(x, q=0.5, epsilon=epsilon, a=a, b=b, seed=seed)
    return RichResult(
        title="DP median",
        summary_lines=[("epsilon", r["epsilon"]), ("release", r["release"])],
        warnings=list(r.warnings),
        payload={
            "release": r["release"], "true_median": r["true_quantile"],
            "interval": r["interval"], "bounds": r["bounds"],
            "n": r["n"], "epsilon": r["epsilon"], "method": "dp_median",
        },
    )


def cheatsheet():
    return "dpmed: global sensitivity of a median is the whole domain -- select on rank, never add noise"
