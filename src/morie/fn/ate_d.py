# morie.fn -- function file (rootcoder007/morie)
"""Average treatment effect under potential outcomes."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["ate_definition"]


def ate_definition(Y1, Y0, paired=True, alpha=0.05):
    r"""Average treatment effect from potential outcomes.

    .. math:: \mathrm{ATE} = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]

    The two forms are equal by linearity of expectation, but their
    *estimators* are not interchangeable, and which one applies depends
    on what the data are.

    When ``Y1`` and ``Y0`` are the two potential outcomes of the **same**
    units -- a simulation, a matched design, or a sensitivity analysis
    where both arms are imputed -- the pairing carries information. The
    variance of the mean difference is then
    :math:`\mathrm{Var}(Y(1) - Y(0))/n`, which accounts for the
    covariance between arms and is usually far smaller than treating
    them as independent.

    When the arms are **different units**, that pairing does not exist
    and the variance is :math:`s_1^2/n_1 + s_0^2/n_0`.

    Getting this wrong is not a rounding error: under a strong positive
    correlation between arms the unpaired standard error can be several
    times too large, and under no correlation the paired one is too
    small. ``paired`` is therefore explicit rather than inferred from
    whether the lengths happen to match.

    This is the *definition*, and it assumes both arms were supplied.
    The fundamental problem of causal inference is that real data never
    show both potential outcomes for the same unit; for an estimate from
    observed data under an identification assumption, use an adjustment
    estimator instead.

    Parameters
    ----------
    Y1, Y0 : array-like
        Potential outcomes under treatment and under control. Equal
        lengths are required when ``paired`` is True.
    paired : bool, default True
        Treat the two arrays as the same units observed under both arms.
    alpha : float, default 0.05
        Significance level for the confidence interval.

    Returns
    -------
    RichResult
        keys: ``ate``, ``estimate``, ``se``, ``ci_low``, ``ci_high``,
        ``statistic`` (t), ``p_value``, ``df``, ``mean_treated``,
        ``mean_control``, ``n1``, ``n0``, ``paired``, ``method``.

    References
    ----------
    Rubin, D. B. (1974). Estimating causal effects of treatments in
    randomized and nonrandomized studies. *Journal of Educational
    Psychology*, 66(5), 688-701.

    Holland, P. W. (1986). Statistics and causal inference. *Journal of
    the American Statistical Association*, 81(396), 945-960.
    """
    a = np.asarray(Y1, dtype=float).ravel()
    b = np.asarray(Y0, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("Y1 and Y0 must not be empty.")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("Y1 and Y0 must be finite.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
    if paired and a.size != b.size:
        raise ValueError(
            f"paired=True needs the same units in both arms; got {a.size} and {b.size}. "
            "Pass paired=False for independent groups."
        )

    ate = float(a.mean() - b.mean())
    if paired:
        d = a - b
        n = d.size
        if n < 2:
            raise ValueError(f"Need at least 2 pairs for a standard error, got {n}.")
        se = float(np.std(d, ddof=1) / np.sqrt(n))
        df = float(n - 1)
    else:
        n1, n0 = a.size, b.size
        if n1 < 2 or n0 < 2:
            raise ValueError(f"Need at least 2 observations per arm, got {n1} and {n0}.")
        v1, v0 = np.var(a, ddof=1) / n1, np.var(b, ddof=1) / n0
        se = float(np.sqrt(v1 + v0))
        # Welch-Satterthwaite: the arms are not assumed equally variable.
        df = float((v1 + v0) ** 2 / (v1**2 / (n1 - 1) + v0**2 / (n0 - 1))) if se > 0 else np.nan

    if se > 0:
        tstat = ate / se
        crit = stats.t.ppf(1 - alpha / 2, df)
        lo, hi = ate - crit * se, ate + crit * se
        pval = float(2 * stats.t.sf(abs(tstat), df))
    else:
        tstat, lo, hi, pval = np.nan, ate, ate, np.nan

    return RichResult(
        title="Average treatment effect",
        payload={
            "ate": ate,
            "estimate": ate,
            "se": se,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "statistic": float(tstat),
            "p_value": pval,
            "df": df,
            "mean_treated": float(a.mean()),
            "mean_control": float(b.mean()),
            "n1": int(a.size),
            "n0": int(b.size),
            "paired": bool(paired),
            "method": "ATE from potential outcomes, " + ("paired" if paired else "independent arms"),
        },
    )


def cheatsheet():
    return "ate_d: average treatment effect under potential outcomes"
