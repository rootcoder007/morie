# morie.fn -- function file (rootcoder007/morie)
"""Balanced repeated replication variance."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["brr_variance"]


def brr_variance(estimates, full_estimate=None, fay_k=0.0):
    r"""Variance of a survey estimate from balanced half-sample replicates.

    .. math::
        \hat V = \frac{1}{R(1-k)^2}\sum_{r=1}^{R}
                  \left(\hat\theta_r - \hat\theta\right)^2 .

    BRR exists because analytic variance formulas are unavailable for most
    real survey statistics -- medians, ratios, model coefficients under
    stratified multistage designs with weighting adjustments. Replicating the
    **whole estimation procedure** on half-samples captures all of it,
    including the weighting steps that an analytic formula would have to
    ignore.

    The Fay adjustment :math:`k` shrinks half-samples toward the full sample
    rather than discarding half the PSUs outright. Its purpose is practical:
    at :math:`k = 0` a replicate can be degenerate -- an empty cell, a model
    that fails to converge -- and :math:`k = 0.3` keeps every unit in every
    replicate with reduced weight, which usually fixes it. The divisor
    :math:`(1-k)^2` is what keeps the variance estimate unbiased under the
    adjustment, and omitting it inflates the variance by a factor of
    :math:`1/(1-k)^2` -- at :math:`k=0.3` that is a 104% overstatement.

    Parameters
    ----------
    estimates : array-like
        Replicate estimates, length ``R``.
    full_estimate : float, optional
        Full-sample estimate. Defaults to the replicate mean.
    fay_k : float
        Fay adjustment in [0, 1).

    Returns
    -------
    RichResult
        ``variance``, ``se``, ``n_replicates``, ``fay_k``, ``cv``.

    References
    ----------
    Judkins, D. R. (1990). Fay's method for variance estimation. *Journal of
        Official Statistics*, 6(3), 223-239.

    Examples
    --------
    Without a Fay adjustment this is the mean squared deviation from the
    full-sample estimate.

    >>> import numpy as np
    >>> reps = np.array([10.0, 12.0, 8.0, 11.0])
    >>> r = brr_variance(reps, full_estimate=10.0)
    >>> float(round(r["variance"], 6))
    2.25

    The Fay divisor matters: omitting it at k = 0.3 would overstate the
    variance by 1/(1-k)^2, slightly more than doubling it.

    >>> f = brr_variance(reps, full_estimate=10.0, fay_k=0.3)
    >>> bool(abs(f["variance"] / r["variance"] - 1 / 0.49) < 1e-9)
    True

    Standard error is the root of the variance, as usual.

    >>> bool(abs(r["se"] - r["variance"] ** 0.5) < 1e-12)
    True

    >>> brr_variance([1.0, 2.0], fay_k=1.0)
    Traceback (most recent call last):
        ...
    ValueError: fay_k must be in [0, 1)
    """
    est = np.atleast_1d(np.asarray(estimates, dtype=float)).ravel()
    if est.size < 2:
        raise ValueError("need at least 2 replicate estimates")
    fay_k = float(fay_k)
    if not 0.0 <= fay_k < 1.0:
        raise ValueError("fay_k must be in [0, 1)")
    theta = float(np.mean(est)) if full_estimate is None else float(full_estimate)
    R = est.size
    var = float(np.sum((est - theta) ** 2) / (R * (1.0 - fay_k) ** 2))
    se = float(np.sqrt(max(var, 0.0)))
    return RichResult(
        title="BRR variance",
        summary_lines=[("replicates", int(R)), ("Fay k", fay_k),
                       ("variance", var), ("se", se)],
        warnings=["the (1-k)^2 divisor is required under a Fay adjustment; "
                  "omitting it inflates the variance by 1/(1-k)^2"],
        payload={
            "variance": var, "se": se, "n_replicates": int(R),
            "fay_k": fay_k, "estimate": theta,
            "cv": float(se / abs(theta)) if theta != 0 else float("nan"),
            "method": "brr_variance",
        },
    )


def cheatsheet():
    return "brrvar: replicates the WHOLE procedure incl. weighting; the (1-k)^2 divisor is mandatory under Fay"
