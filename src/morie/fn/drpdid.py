# morie.fn -- function file (rootcoder007/morie)
"""Placebo doubly-robust difference-in-differences."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from .drovw import dr_overlap_weighted

__all__ = ["placebo_dr_did"]


def placebo_dr_did(y_pre1, y_pre2, D, X, **kwargs):
    r"""Run the DiD estimator on two pre-treatment periods, where the effect is zero.

    Applies the doubly-robust estimator to the change between two periods that
    both precede treatment. Since nothing happened, the estimate should be
    zero; a non-zero one is direct evidence that the parallel-trends
    assumption fails for these groups.

    This is the strongest routinely available check on DiD, because it tests
    the identifying assumption on the same machinery used for the real
    estimate -- the same covariates, the same propensity model, the same
    weighting. A pre-trend test run with a different specification tests a
    different assumption than the one the headline estimate relies on.

    The usual caution applies in the same form as for any falsification test:
    failure is informative, passing is weak. ``min_detectable`` is returned so
    a null can be read against what the test could have found, and a
    pre-trend smaller than the effect of interest is not evidence of parallel
    trends.

    Parameters
    ----------
    y_pre1, y_pre2 : array-like
        Outcome in two pre-treatment periods.
    D : array-like
        Eventual treatment indicator, 0/1.
    X : array-like
        Covariates.
    **kwargs
        Passed to :func:`~morie.fn.drovw.dr_overlap_weighted`.

    Returns
    -------
    RichResult
        ``placebo_effect``, ``se``, ``ci``, ``passed``, ``min_detectable``.

    References
    ----------
    Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust
        difference-in-differences estimators. *Journal of Econometrics*,
        219(1), 101-122.

    Examples
    --------
    With genuinely parallel pre-trends the placebo effect is near zero.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(3000, 2))
    >>> e = 1 / (1 + np.exp(-(0.7 * X[:, 0])))
    >>> D = (rng.random(3000) < e).astype(float)
    >>> base = X[:, 0] + rng.normal(0, 0.4, 3000)
    >>> r = placebo_dr_did(base, base + rng.normal(0, 0.4, 3000), D, X)
    >>> bool(r["passed"])
    True

    A differential pre-trend is caught, which is the informative direction.

    >>> bad = base + 1.5 * D + rng.normal(0, 0.4, 3000)
    >>> bool(not placebo_dr_did(base, bad, D, X)["passed"])
    True

    The minimum detectable effect is reported so a null can be read in
    proportion.

    >>> bool(r["min_detectable"] > 0)
    True
    """
    y1 = np.atleast_1d(np.asarray(y_pre1, dtype=float)).ravel()
    y2 = np.atleast_1d(np.asarray(y_pre2, dtype=float)).ravel()
    if y1.size != y2.size:
        raise ValueError("the two pre-period outcomes must have the same length")
    r = dr_overlap_weighted(y2 - y1, D, X, **kwargs)
    est, se = float(r["ate"]), float(r["se"])
    z = est / se if se > 0 else np.nan
    from scipy.stats import norm

    p = float(2 * norm.sf(abs(z))) if se > 0 else float("nan")
    return RichResult(
        title="Placebo DR difference-in-differences",
        summary_lines=[("placebo effect", est), ("se", se), ("p", p)],
        warnings=["failure is informative, passing is weak: a pre-trend "
                  "smaller than the effect of interest is not evidence of "
                  "parallel trends"],
        payload={
            "placebo_effect": est, "se": se, "z": float(z), "p_value": p,
            "ci": r["ci"], "passed": bool(p > 0.05) if np.isfinite(p) else False,
            "min_detectable": float(2.8 * se),
            "propensity": r["propensity"], "method": "placebo_dr_did",
        },
    )


def cheatsheet():
    return "drpdid: tests parallel trends with the SAME machinery as the headline estimate; failure informative"
