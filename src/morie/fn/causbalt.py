# morie.fn -- function file (rootcoder007/morie)
"""Standardised mean difference balance test after weighting."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["causal_balance_test"]


def _wmean_wvar(x, w):
    """Weighted mean and the weighted variance used for the SMD."""
    sw = w.sum()
    m = float(np.sum(w * x) / sw)
    # sw - sum(w^2)/sw is the standard reliability correction for a
    # weighted variance. It reduces to n - 1 when the weights are equal,
    # so the weighted and unweighted diagnostics agree in that case.
    denom = sw - np.sum(w**2) / sw
    v = float(np.sum(w * (x - m) ** 2) / denom) if denom > 0 else np.nan
    return m, v


def causal_balance_test(X, treat, weights=None, cdf=None, threshold=0.1):
    r"""Standardised mean difference for covariate balance.

    For each column of ``X`` the standardised mean difference is

    .. math::

        \mathrm{SMD} = \frac{\bar\mu_t - \bar\mu_c}
                            {\sqrt{(s_t^2 + s_c^2)/2}}

    the difference in group means in units of the pooled within-group
    spread. The pooling is a plain average of the two variances rather
    than the sample-size-weighted one, which is deliberate: the
    denominator then does not move when weighting changes the effective
    group sizes, so the same covariate stays comparable before and after
    weighting.

    With ``weights`` the means and variances are weighted, which is the
    balance diagnostic for an inverse-probability-weighted sample.

    No p-value is reported by default, and that is the point. Austin's
    argument is that a hypothesis test of balance confounds imbalance
    with sample size, so a test can be passed simply by having few
    subjects. The convention is to compare |SMD| against a fixed
    threshold, 0.1 by default. Pass ``cdf`` if a p-value on the largest
    |SMD| is genuinely wanted.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Covariates. A one-dimensional input is read as a single column.
    treat : array-like, shape (n,)
        Binary treatment indicator. Exactly two distinct values; the
        one sorting second is treated as "treated".
    weights : array-like, optional
        Non-negative observation weights. Defaults to equal weights.
    cdf : callable, optional
        Null CDF for the maximum |SMD|, if a p-value is wanted.
    threshold : float, default 0.1
        Imbalance cutoff for |SMD|.

    Returns
    -------
    RichResult
        keys: ``smd`` (per column), ``max_smd``, ``imbalanced`` (column
        indices over threshold), ``n_imbalanced``, ``balanced``,
        ``threshold``, ``n_treated``, ``n_control``, ``p_value``,
        ``method``.

    References
    ----------
    Austin, P. C. (2009). Balance diagnostics for comparing the
    distribution of baseline covariates between treatment groups in
    propensity-score matched samples. *Statistics in Medicine*, 28(25),
    3083-3107.
    """
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    if Xa.ndim != 2:
        raise ValueError(f"X must be (n, p); got shape {Xa.shape}.")
    t = np.asarray(treat).ravel()
    n, p = Xa.shape
    if t.size != n:
        raise ValueError(f"treat must have one entry per row of X; got {t.size} and {n}.")
    levels = np.unique(t)
    if levels.size != 2:
        raise ValueError(f"treat must be binary; got {levels.size} distinct values.")
    if not np.all(np.isfinite(Xa)):
        raise ValueError("X must be finite.")

    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.size != n:
        raise ValueError(f"weights must have one entry per row of X; got {w.size} and {n}.")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative.")

    is_t = t == levels[1]
    if w[is_t].sum() <= 0 or w[~is_t].sum() <= 0:
        raise ValueError("Both treatment groups need positive total weight.")

    smd = np.empty(p)
    for j in range(p):
        mt, vt = _wmean_wvar(Xa[is_t, j], w[is_t])
        mc, vc = _wmean_wvar(Xa[~is_t, j], w[~is_t])
        pooled = (vt + vc) / 2.0
        smd[j] = (mt - mc) / np.sqrt(pooled) if pooled > 0 else 0.0

    absmax = float(np.max(np.abs(smd)))
    over = np.flatnonzero(np.abs(smd) > threshold)

    return RichResult(
        title="Covariate balance (standardised mean differences)",
        payload={
            "smd": smd,
            "max_smd": absmax,
            "imbalanced": over,
            "n_imbalanced": int(over.size),
            "balanced": bool(over.size == 0),
            "threshold": float(threshold),
            "n_treated": int(np.sum(is_t)),
            "n_control": int(np.sum(~is_t)),
            "p_value": float(1.0 - cdf(absmax)) if cdf is not None else None,
            "method": "Standardised mean difference (Austin 2009)" + (", weighted" if weights is not None else ""),
        },
    )


def cheatsheet():
    return "causbalt: standardised mean difference balance diagnostic"
