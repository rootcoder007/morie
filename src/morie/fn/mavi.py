# morie.fn -- function file (rootcoder007/morie)
"""Variance inflation for correlated effect sizes (Hedges, Tipton & Johnson 2010)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ma_var_inflation_correlated"]


def ma_var_inflation_correlated(V, rho):
    r"""Inflate sampling variances to account for within-cluster correlation.

    For effect sizes sharing a cluster with common correlation :math:`\rho`,
    the variance of their sum picks up every covariance term:

    .. math::

        V^{*} = \sum_i V_i + 2\sum_{i}\sum_{j>i} \operatorname{cov}_{ij},
        \qquad \operatorname{cov}_{ij} = \rho\sqrt{V_i V_j}.

    Parameters
    ----------
    V : array-like, shape (k,)
        Sampling variances of the ``k`` dependent effect sizes. Must be
        non-negative -- these are variances, not standard errors.
    rho : float
        Assumed correlation between effect sizes in the cluster,
        :math:`-1 \le \rho \le 1`.

    Returns
    -------
    RichResult
        keys: ``V_inflated`` (variance of the *sum*), ``V_mean_inflated``
        (variance of the unweighted *mean*, the quantity a synthesis usually
        wants), ``V_naive`` (the sum ignoring covariance), ``inflation_factor``,
        ``k``, ``rho``, ``method``.

    Raises
    ------
    ValueError
        If any variance is negative, or ``rho`` is outside :math:`[-1, 1]`,
        or the implied covariance matrix is not positive semi-definite.

    References
    ----------
    Hedges, L. V., Tipton, E., & Johnson, M. C. (2010). Robust variance
        estimation in meta-regression with dependent effect size estimates.
        *Research Synthesis Methods*, 1(1), 39-65.

    Notes
    -----
    Treating dependent effect sizes as independent understates the variance
    whenever :math:`\rho > 0`, which is the usual case -- multiple outcomes
    measured on one sample are positively correlated. With ``k`` equal
    variances the inflation factor is :math:`1 + (k-1)\rho`, so five outcomes
    at :math:`\rho = 0.5` carry **three times** the naive variance, and a
    confidence interval built on the naive value is 42% too narrow.

    A constant-:math:`\rho` (compound symmetric) matrix is only PSD for
    :math:`\rho \ge -1/(k-1)`. Values below that describe no real covariance
    structure, so they raise rather than returning a negative variance.

    Both the sum and the mean are returned because the distinction is where
    this is usually got wrong: the variance of the *mean* is
    :math:`V^{*}/k^2`, not :math:`V^{*}/k`.
    """
    v = np.asarray(V, dtype=float).ravel()
    rho = float(rho)
    k = v.size
    if k < 1:
        raise ValueError("V must contain at least one variance")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"V must be finite; got {V!r}")
    if np.any(v < 0):
        raise ValueError(
            f"variances must be non-negative; got {V!r}. These are variances, "
            "not standard errors."
        )
    if not (-1.0 <= rho <= 1.0):
        raise ValueError(f"rho must lie in [-1, 1]; got {rho!r}")
    if k > 1 and rho < -1.0 / (k - 1):
        raise ValueError(
            f"rho={rho!r} with k={k} gives a compound-symmetric covariance matrix "
            f"that is not positive semi-definite (requires rho >= {-1.0 / (k - 1)!r}). "
            "No set of random variables has that correlation structure."
        )
    naive = float(v.sum())
    sd = np.sqrt(v)
    # 2 * sum_{j>i} rho sqrt(Vi Vj) = rho * [(sum sd)^2 - sum V]
    cross = rho * (float(sd.sum()) ** 2 - naive)
    inflated = naive + cross
    return RichResult(
        payload={
            "V_inflated": inflated,
            "V_mean_inflated": inflated / (k**2),
            "V_naive": naive,
            "inflation_factor": (inflated / naive) if naive > 0 else float("nan"),
            "k": int(k),
            "rho": rho,
            "method": "variance inflation for correlated effects (Hedges, Tipton & Johnson 2010)",
        }
    )


def cheatsheet():
    return "mavi: V* = sum V_i + 2 sum_{j>i} rho sqrt(V_i V_j) (Hedges-Tipton-Johnson 2010)."
