"""
Comprehensive effect-size calculations for epidemiological research.

This module provides every major family of effect-size estimators used in
biomedical and social-science research, each with analytic or bootstrap
confidence intervals.  Functions are organised by the type of comparison:

- **Standardised mean differences**: Cohen's *d*, Hedges' *g*, Glass's delta
- **Common-language effect sizes**: CLES / probability of superiority
- **Correlation-based**: *r*, *R*-squared, eta-squared, partial eta-squared,
  omega-squared, epsilon-squared
- **Contingency-table measures**: odds ratio, risk ratio, risk difference, NNT,
  NNH, rate ratio, incidence rate difference
- **Association measures**: Cohen's *w*, Cramer's *V*, phi coefficient
- **Non-parametric**: rank-biserial correlation, Cliff's delta, Vargha--Delaney *A*
- **Regression**: standardised coefficients, coefficient of variation
- **Conversion**: *d* <-> *r*, OR <-> *d*, etc.
- **Meta-analysis**: fixed- and random-effects pooling, *I*-squared,
  prediction intervals

References
----------
Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
    (2nd ed.). Lawrence Erlbaum Associates.
Hedges, L. V., & Olkin, I. (1985). *Statistical Methods for Meta-Analysis*.
    Academic Press.
Borenstein, M., Hedges, L. V., Higgins, J. P. T., & Rothstein, H. R. (2009).
    *Introduction to Meta-Analysis*. Wiley.
Vargha, A., & Delaney, H. D. (2000). A critique and improvement of the CL
    common language effect size statistics of McGraw and Wong. *JEBS*, 25(2),
    101--132.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Union

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn import _stats_core as stats

logger = logging.getLogger(__name__)


# ===================================================================
# Result container
# ===================================================================


@dataclass
class EffectSizeResult:
    """Standardised result for every effect-size calculation.

    Parameters
    ----------
    measure : str
        Name of the effect-size statistic.
    estimate : float
        Point estimate.
    ci_lower : float | None
        Lower confidence bound.
    ci_upper : float | None
        Upper confidence bound.
    se : float | None
        Standard error (analytic or bootstrap).
    n : int | None
        Sample size used.
    extra : dict
        Additional outputs.
    """

    measure: str
    estimate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    se: float | None = None
    n: int | None = None
    extra: dict = field(default_factory=dict)


# ===================================================================
# Helpers
# ===================================================================


def _arr(x: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    """Coerce to float64 array, drop NaN."""
    a = np.asarray(x, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def _bootstrap_ci(
    func,
    args: tuple,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI and SE for an effect-size function.

    Parameters
    ----------
    func : callable
        Function that takes resampled arrays and returns a scalar.
    args : tuple of arrays
        Original arrays to bootstrap.
    n_boot : int
    confidence : float
    seed : int

    Returns
    -------
    tuple
        (se, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    boot_vals = np.empty(n_boot)
    arrays = [np.asarray(a) for a in args]
    for b in range(n_boot):
        resampled = tuple(a[rng.choice(len(a), len(a), replace=True)] for a in arrays)
        try:
            boot_vals[b] = func(*resampled)
        except Exception:
            boot_vals[b] = np.nan
    boot_vals = boot_vals[np.isfinite(boot_vals)]
    if len(boot_vals) == 0:
        return 0.0, np.nan, np.nan
    alpha = (1 - confidence) / 2
    ci_lo = float(np.percentile(boot_vals, 100 * alpha))
    ci_hi = float(np.percentile(boot_vals, 100 * (1 - alpha)))
    se = float(np.std(boot_vals, ddof=1))
    return se, ci_lo, ci_hi


# ===================================================================
# STANDARDISED MEAN DIFFERENCES
# ===================================================================


def cohens_d(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Cohen's *d* for independent samples (pooled SD denominator).

    .. math::
        d = \\frac{\\bar{x} - \\bar{y}}{s_p}

    Parameters
    ----------
    x, y : array-like
        Two independent samples.
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    nx, ny = len(x), len(y)
    sp = math.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    d = (x.mean() - y.mean()) / sp if sp > 0 else 0.0
    # Hedges & Olkin (1985, p. 86) large-sample variance, as metafor::escalc
    se = math.sqrt((nx + ny) / (nx * ny) + d**2 / (2 * (nx + ny)))
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Cohen's d",
        estimate=float(d),
        ci_lower=float(d - z * se),
        ci_upper=float(d + z * se),
        se=float(se),
        n=nx + ny,
    )


def hedges_g(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Hedges' *g* -- bias-corrected Cohen's *d*.

    Applies the exact correction factor
    :math:`J(m) = \\Gamma(m/2) / (\\sqrt{m/2}\\,\\Gamma((m-1)/2))`, :math:`m = n_1+n_2-2`
    (Hedges 1981); :math:`1 - 3/(4m-1)` is its large-*m* approximation.

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    d_result = cohens_d(x, y, confidence)
    df_val = len(x) + len(y) - 2
    J = (
        math.exp(math.lgamma(df_val / 2) - 0.5 * math.log(df_val / 2) - math.lgamma((df_val - 1) / 2))
        if df_val > 1
        else 1.0
    )
    g = d_result.estimate * J
    se = d_result.se * J if d_result.se else 0.0
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Hedges' g",
        estimate=float(g),
        ci_lower=float(g - z * se),
        ci_upper=float(g + z * se),
        se=float(se),
        n=d_result.n,
        extra={"correction_factor": float(J)},
    )


def glass_delta(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    control: str = "y",
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Glass's delta -- uses the control group SD as denominator.

    Parameters
    ----------
    x, y : array-like
    control : str, default "y"
        Which group is the control: ``"x"`` or ``"y"``.
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    ctrl = y if control == "y" else x
    sd_ctrl = ctrl.std(ddof=1)
    delta = (x.mean() - y.mean()) / sd_ctrl if sd_ctrl > 0 else 0.0
    n_ctrl = len(ctrl)
    se = math.sqrt(1 / len(x) + 1 / len(y) + delta**2 / (2 * (n_ctrl - 1)))
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Glass's delta",
        estimate=float(delta),
        ci_lower=float(delta - z * se),
        ci_upper=float(delta + z * se),
        se=float(se),
        n=len(x) + len(y),
    )


# ===================================================================
# COMMON LANGUAGE EFFECT SIZE
# ===================================================================


def cles(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Common Language Effect Size (probability of superiority).

    Estimates :math:`P(X > Y)` for randomly drawn observations from each group.

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    nx, ny = len(x), len(y)

    def psup(a, b):
        dm = np.subtract.outer(a, b)
        return (float((dm > 0).sum()) + 0.5 * float((dm == 0).sum())) / (len(a) * len(b))

    p_sup = psup(x, y) if nx * ny > 0 else 0.5
    se, ci_lo, ci_hi = _bootstrap_ci(
        psup,
        (x, y),
        confidence=confidence,
    )
    return EffectSizeResult(
        measure="CLES (Prob. of superiority)",
        estimate=float(p_sup),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=nx + ny,
    )


# ===================================================================
# CORRELATION-BASED
# ===================================================================


def r_effect_size(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Pearson *r* as an effect size with Fisher *z* CI.

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    r, _ = stats.pearsonr(x, y)
    z_r = np.arctanh(r)
    se_z = 1 / math.sqrt(n - 3) if n > 3 else np.inf
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Pearson r",
        estimate=float(r),
        ci_lower=float(np.tanh(z_r - z_crit * se_z)),
        ci_upper=float(np.tanh(z_r + z_crit * se_z)),
        se=float(se_z),
        n=n,
    )


def r_squared(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
) -> EffectSizeResult:
    """Coefficient of determination *R*-squared.

    Parameters
    ----------
    x, y : array-like

    Returns
    -------
    EffectSizeResult
    """
    r_res = r_effect_size(x, y)
    r2 = r_res.estimate**2
    # r^2 is not monotone in r: an r interval that spans 0 maps to
    # [0, max(lo^2, hi^2)], a negative one to [hi^2, lo^2]
    lo, hi = r_res.ci_lower, r_res.ci_upper
    if lo is None or hi is None:
        ci = (None, None)
    elif lo >= 0:
        ci = (float(lo**2), float(hi**2))
    elif hi <= 0:
        ci = (float(hi**2), float(lo**2))
    else:
        ci = (0.0, float(max(lo**2, hi**2)))
    return EffectSizeResult(
        measure="R-squared",
        estimate=float(r2),
        ci_lower=ci[0],
        ci_upper=ci[1],
        n=r_res.n,
    )


def eta_squared(
    ss_effect: float,
    ss_total: float,
) -> EffectSizeResult:
    """Eta-squared from ANOVA sums of squares.

    :math:`\\eta^2 = SS_{effect} / SS_{total}`

    Parameters
    ----------
    ss_effect : float
        Sum of squares for the effect.
    ss_total : float
        Total sum of squares.

    Returns
    -------
    EffectSizeResult
    """
    eta2 = ss_effect / ss_total if ss_total > 0 else 0.0
    return EffectSizeResult(
        measure="Eta-squared",
        estimate=float(eta2),
    )


def partial_eta_squared(
    ss_effect: float,
    ss_error: float,
) -> EffectSizeResult:
    """Partial eta-squared.

    :math:`\\eta^2_p = SS_{effect} / (SS_{effect} + SS_{error})`

    Parameters
    ----------
    ss_effect, ss_error : float

    Returns
    -------
    EffectSizeResult
    """
    denom = ss_effect + ss_error
    pe2 = ss_effect / denom if denom > 0 else 0.0
    return EffectSizeResult(
        measure="Partial eta-squared",
        estimate=float(pe2),
    )


def omega_squared(
    ss_effect: float,
    ss_total: float,
    df_effect: int,
    ms_error: float,
) -> EffectSizeResult:
    """Omega-squared -- less biased than eta-squared.

    :math:`\\omega^2 = (SS_{effect} - df_{effect} \\cdot MS_{error}) / (SS_{total} + MS_{error})`

    Parameters
    ----------
    ss_effect, ss_total : float
    df_effect : int
    ms_error : float

    Returns
    -------
    EffectSizeResult
    """
    num = ss_effect - df_effect * ms_error
    denom = ss_total + ms_error
    w2 = max(num / denom, 0.0) if denom > 0 else 0.0
    return EffectSizeResult(
        measure="Omega-squared",
        estimate=float(w2),
    )


def epsilon_squared(
    ss_effect: float,
    ss_total: float,
    df_effect: int,
    ms_error: float,
) -> EffectSizeResult:
    """Epsilon-squared (Kelley, 1935).

    :math:`\\varepsilon^2 = (SS_{effect} - df_{effect} \\cdot MS_{error}) / SS_{total}`

    Parameters
    ----------
    ss_effect, ss_total : float
    df_effect : int
    ms_error : float

    Returns
    -------
    EffectSizeResult
    """
    num = ss_effect - df_effect * ms_error
    eps2 = max(num / ss_total, 0.0) if ss_total > 0 else 0.0
    return EffectSizeResult(
        measure="Epsilon-squared",
        estimate=float(eps2),
    )


# ===================================================================
# CONTINGENCY TABLE EFFECT SIZES
# ===================================================================


def odds_ratio(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Odds ratio for a 2x2 table [[a, b], [c, d]].

    :math:`OR = (a \\cdot d) / (b \\cdot c)`

    Parameters
    ----------
    a, b, c, d : int
        Cell counts of the 2x2 table.
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    # Haldane-Anscombe: with any zero cell, 1/2 is added to all four, as
    # metafor::escalc(measure="OR") does by default (add=1/2, to="only0")
    n = a + b + c + d
    cc = 0.5 if min(a, b, c, d) == 0 else 0.0
    a, b, c, d = a + cc, b + cc, c + cc, d + cc
    or_val = (a * d) / (b * c)
    log_or = math.log(or_val)
    se_log = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Odds ratio",
        estimate=float(or_val),
        ci_lower=float(math.exp(log_or - z * se_log)),
        ci_upper=float(math.exp(log_or + z * se_log)),
        se=float(se_log),
        n=n,
        extra={"log_or": log_or, "continuity_correction": cc},
    )


def risk_ratio(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Risk ratio (relative risk) for a 2x2 table.

    :math:`RR = [a/(a+b)] / [c/(c+d)]`

    Parameters
    ----------
    a, b, c, d : int
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    # 1/2 added to all four cells when any is zero (metafor::escalc "RR")
    n = a + b + c + d
    cc = 0.5 if min(a, b, c, d) == 0 else 0.0
    a, b, c, d = a + cc, b + cc, c + cc, d + cc
    rr = (a / (a + b)) / (c / (c + d))
    log_rr = math.log(rr)
    se_log = math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Risk ratio",
        estimate=float(rr),
        ci_lower=float(math.exp(log_rr - z * se_log)),
        ci_upper=float(math.exp(log_rr + z * se_log)),
        se=float(se_log),
        n=n,
        extra={"continuity_correction": cc},
    )


def risk_difference(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Risk difference (attributable risk) for a 2x2 table.

    :math:`RD = a/(a+b) - c/(c+d)`

    Parameters
    ----------
    a, b, c, d : int
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    n1, n2 = a + b, c + d
    p1 = a / n1 if n1 > 0 else 0.0
    p2 = c / n2 if n2 > 0 else 0.0
    rd = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if n1 > 0 and n2 > 0 else 0.0
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Risk difference",
        estimate=float(rd),
        ci_lower=float(rd - z * se),
        ci_upper=float(rd + z * se),
        se=float(se),
        n=n1 + n2,
    )


def number_needed_to_treat(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Number needed to treat (NNT).

    :math:`NNT = 1 / |RD|`

    Parameters
    ----------
    a, b, c, d : int
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    rd_res = risk_difference(a, b, c, d, confidence)
    rd = rd_res.estimate
    nnt = 1 / abs(rd) if abs(rd) > 0 else np.inf
    lo, hi = rd_res.ci_lower, rd_res.ci_upper
    # Altman (1998): when the RD interval spans 0 the NNT interval is
    # disjoint, from one bound through infinity to the other; ci_lower is
    # then the smaller finite limit and ci_upper is inf
    spans = lo is not None and hi is not None and lo < 0 < hi
    lim = [1 / abs(v) if v else np.inf for v in (lo, hi)]
    ci = (min(lim), np.inf) if spans else tuple(sorted(lim))
    return EffectSizeResult(
        measure="NNT",
        estimate=float(nnt),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        n=rd_res.n,
        extra={"ci_spans_zero": spans},
    )


def number_needed_to_harm(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Number needed to harm (NNH) -- same as NNT but with reversed sign
    convention.

    Parameters
    ----------
    a, b, c, d : int
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    result = number_needed_to_treat(a, b, c, d, confidence)
    return EffectSizeResult(
        measure="NNH",
        estimate=result.estimate,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        n=result.n,
    )


def rate_ratio(
    events1: int,
    person_time1: float,
    events2: int,
    person_time2: float,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Incidence rate ratio.

    :math:`IRR = (e_1/PT_1) / (e_2/PT_2)`

    Parameters
    ----------
    events1, person_time1 : int, float
        Events and person-time in group 1.
    events2, person_time2 : int, float
        Events and person-time in group 2.
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    # 1/2 added to both event counts when either is zero (metafor::escalc "IRR")
    cc = 0.5 if min(events1, events2) == 0 else 0.0
    e1, e2 = events1 + cc, events2 + cc
    irr = (e1 / person_time1) / (e2 / person_time2)
    log_irr = math.log(irr)
    se = math.sqrt(1 / e1 + 1 / e2)
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Rate ratio",
        estimate=float(irr),
        ci_lower=float(math.exp(log_irr - z * se)),
        ci_upper=float(math.exp(log_irr + z * se)),
        se=float(se),
        n=events1 + events2,
        extra={"continuity_correction": cc},
    )


def incidence_rate_difference(
    events1: int,
    person_time1: float,
    events2: int,
    person_time2: float,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Incidence rate difference.

    :math:`IRD = (e_1/PT_1) - (e_2/PT_2)`

    Parameters
    ----------
    events1, person_time1 : int, float
    events2, person_time2 : int, float
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    r1 = events1 / person_time1 if person_time1 > 0 else 0.0
    r2 = events2 / person_time2 if person_time2 > 0 else 0.0
    ird = r1 - r2
    se = (
        math.sqrt(events1 / person_time1**2 + events2 / person_time2**2)
        if person_time1 > 0 and person_time2 > 0
        else 0.0
    )
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Incidence rate difference",
        estimate=float(ird),
        ci_lower=float(ird - z * se),
        ci_upper=float(ird + z * se),
        se=float(se),
    )


# ===================================================================
# ASSOCIATION MEASURES
# ===================================================================


def cohens_w(
    observed: Union[np.ndarray, list],
    expected: Union[np.ndarray, list] | None = None,
) -> EffectSizeResult:
    """Cohen's *w* for chi-squared.

    :math:`w = \\sqrt{\\chi^2 / N}`

    Parameters
    ----------
    observed : array-like
        Observed frequencies.
    expected : array-like or None
        Expected frequencies (uniform if ``None``).

    Returns
    -------
    EffectSizeResult
    """
    obs = np.asarray(observed, dtype=np.float64)
    exp = np.full_like(obs, obs.sum() / len(obs)) if expected is None else np.asarray(expected, dtype=np.float64)
    n = obs.sum()
    chi2 = np.sum((obs - exp) ** 2 / (exp + 1e-15))
    w = math.sqrt(chi2 / n) if n > 0 else 0.0
    return EffectSizeResult(
        measure="Cohen's w",
        estimate=float(w),
        n=int(n),
    )


def cohens_f(
    eta2: float,
) -> EffectSizeResult:
    """Cohen's *f* from eta-squared.

    :math:`f = \\sqrt{\\eta^2 / (1 - \\eta^2)}`

    Parameters
    ----------
    eta2 : float
        Eta-squared value.

    Returns
    -------
    EffectSizeResult
    """
    f_val = math.sqrt(eta2 / (1 - eta2)) if eta2 < 1 else np.inf
    return EffectSizeResult(
        measure="Cohen's f",
        estimate=float(f_val),
    )


def cramers_v(
    contingency_table: Union[np.ndarray, pd.DataFrame],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Cramer's *V* for a contingency table.

    :math:`V = \\sqrt{\\chi^2 / (N \\cdot (k - 1))}` where :math:`k = \\min(r, c)`.

    Parameters
    ----------
    contingency_table : array-like or DataFrame
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    table = np.asarray(contingency_table, dtype=np.float64)
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    k = min(table.shape) - 1
    v = math.sqrt(chi2 / (n * k)) if n * k > 0 else 0.0
    # Bias-corrected V (Bergsma 2013, eq. 4-5): phi^2 less its bias, over
    # the bias-corrected table dimensions
    r, c = table.shape
    phi2c = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
    kc = min(r - (r - 1) ** 2 / (n - 1), c - (c - 1) ** 2 / (n - 1)) - 1
    v_bc = math.sqrt(phi2c / kc) if phi2c > 0 and kc > 0 else 0.0
    # confidence interval by inverting the noncentral chi-square: the
    # noncentrality lambda = n k V^2 (Smithson 2003), so the bounds on
    # lambda give bounds on V
    dof = (r - 1) * (c - 1)
    lo_l, hi_l = _ncp_interval(float(chi2), int(dof), float(confidence))
    ci_lower = math.sqrt(lo_l / (n * k)) if n * k > 0 else 0.0
    ci_upper = math.sqrt(hi_l / (n * k)) if n * k > 0 else 0.0
    return EffectSizeResult(
        measure="Cramer's V",
        estimate=float(v),
        ci_lower=float(ci_lower),
        ci_upper=float(min(ci_upper, 1.0)),
        n=int(n),
        extra={
            "bias_corrected_v": float(v_bc),
            "confidence": float(confidence),
            "ci_method": "noncentral chi-square inversion",
        },
    )


def _ncp_interval(chi2_obs, dof, confidence):
    """Bounds on the noncentrality parameter of a chi-square statistic:
    the lambda at which the observed value sits at the (1+c)/2 and (1-c)/2
    quantiles of ncx2(dof, lambda)."""
    from morie.fn._stats_core import ncx2

    def solve(target):
        if ncx2.cdf(chi2_obs, dof, 0.0) <= target:
            return 0.0
        lo, hi = 0.0, max(chi2_obs, 1.0) * 4.0 + 10.0
        while ncx2.cdf(chi2_obs, dof, hi) > target:
            hi *= 2.0
            if hi > 1e7:
                break
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if ncx2.cdf(chi2_obs, dof, mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    a = 1.0 - confidence
    return solve(1.0 - a / 2.0), solve(a / 2.0)


def phi_coefficient(
    contingency_table: Union[np.ndarray, pd.DataFrame],
) -> EffectSizeResult:
    """Phi coefficient for a 2x2 contingency table.

    :math:`\\phi = \\sqrt{\\chi^2 / N}`

    Parameters
    ----------
    contingency_table : array-like

    Returns
    -------
    EffectSizeResult
    """
    table = np.asarray(contingency_table, dtype=np.float64)
    if table.shape != (2, 2):
        raise ValueError("Phi requires a 2x2 table.")
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    phi = math.sqrt(chi2 / n) if n > 0 else 0.0
    # Sign from the OR direction
    if table[0, 0] * table[1, 1] < table[0, 1] * table[1, 0]:
        phi = -phi
    return EffectSizeResult(
        measure="Phi coefficient",
        estimate=float(phi),
        n=int(n),
    )


# ===================================================================
# NON-PARAMETRIC EFFECT SIZES
# ===================================================================


def rank_biserial_correlation(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Rank-biserial correlation for two independent samples.

    Glass (1965): positive when *x* tends to exceed *y* (equal to Cliff's delta).

    :math:`r = 2U / (n_1 n_2) - 1` where *U* is the Mann--Whitney statistic of *x*.

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    nx, ny = len(x), len(y)
    r = 2 * u / (nx * ny) - 1 if nx * ny > 0 else 0.0
    se, ci_lo, ci_hi = _bootstrap_ci(
        lambda a, b: 2 * stats.mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b)) - 1,
        (x, y),
        confidence=confidence,
    )
    return EffectSizeResult(
        measure="Rank-biserial correlation",
        estimate=float(r),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=nx + ny,
    )


def cliffs_delta(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Cliff's delta (non-parametric effect size).

    :math:`\\delta = (\\#(x_i > y_j) - \\#(x_i < y_j)) / (n_x n_y)`

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    nx, ny = len(x), len(y)
    greater = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                greater += 1
            elif xi < yj:
                less += 1
    delta = (greater - less) / (nx * ny) if nx * ny > 0 else 0.0

    def _delta(a, b):
        g = sum(1 for ai in a for bj in b if ai > bj)
        l_ = sum(1 for ai in a for bj in b if ai < bj)
        return (g - l_) / (len(a) * len(b))

    se, ci_lo, ci_hi = _bootstrap_ci(_delta, (x, y), confidence=confidence)
    return EffectSizeResult(
        measure="Cliff's delta",
        estimate=float(delta),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=nx + ny,
    )


def vargha_delaney_a(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Vargha--Delaney *A* statistic.

    :math:`A = U / (n_1 n_2)` where *U* is the Mann--Whitney statistic
    (probability that a randomly chosen *x* exceeds a randomly chosen *y*).

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    nx, ny = len(x), len(y)
    a_val = u / (nx * ny) if nx * ny > 0 else 0.5
    se, ci_lo, ci_hi = _bootstrap_ci(
        lambda a, b: stats.mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b)),
        (x, y),
        confidence=confidence,
    )
    return EffectSizeResult(
        measure="Vargha-Delaney A",
        estimate=float(a_val),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=nx + ny,
    )


# ===================================================================
# REGRESSION
# ===================================================================


def standardized_coefficients(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
) -> pd.DataFrame:
    """Compute standardised regression coefficients (beta weights).

    Standardises X and y to zero mean and unit variance before OLS.

    Parameters
    ----------
    X : array-like or DataFrame
        Predictor matrix (n x p).
    y : array-like or Series
        Outcome variable.

    Returns
    -------
    DataFrame
        Columns: ``variable``, ``beta``, ``se``, ``t``, ``p_value``.
    """
    from morie.fn import _glm_core as sm

    if isinstance(X, pd.DataFrame):
        names = X.columns.tolist()
        X_arr = X.values.astype(np.float64)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        names = [f"x{i}" for i in range(X_arr.shape[1])]
    y_arr = np.asarray(y, dtype=np.float64).ravel()

    # Standardise
    X_std = (X_arr - X_arr.mean(axis=0)) / (X_arr.std(axis=0, ddof=1) + 1e-15)
    y_std = (y_arr - y_arr.mean()) / (y_arr.std(ddof=1) + 1e-15)

    model = sm.OLS(y_std, sm.add_constant(X_std)).fit()
    # Skip constant (index 0)
    results = []
    for i, name in enumerate(names):
        results.append(
            {
                "variable": name,
                "beta": float(model.params[i + 1]),
                "se": float(model.bse[i + 1]),
                "t": float(model.tvalues[i + 1]),
                "p_value": float(model.pvalues[i + 1]),
            }
        )
    return pd.DataFrame(results)


def coefficient_of_variation(
    x: Union[np.ndarray, pd.Series, list],
) -> EffectSizeResult:
    """Coefficient of variation (CV).

    :math:`CV = s / \\bar{x}`

    Parameters
    ----------
    x : array-like

    Returns
    -------
    EffectSizeResult
    """
    x = _arr(x)
    mean = x.mean()
    sd = x.std(ddof=1)
    cv = sd / abs(mean) if abs(mean) > 0 else np.inf
    return EffectSizeResult(
        measure="Coefficient of variation",
        estimate=float(cv),
        n=len(x),
    )


def variance_ratio(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Variance ratio (*F*-test for equality of variances).

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    f_val = v1 / v2 if v2 > 0 else np.inf
    df1, df2 = len(x) - 1, len(y) - 1
    alpha = (1 - confidence) / 2
    ci_lo = f_val / stats.f.ppf(1 - alpha, df1, df2)
    ci_hi = f_val / stats.f.ppf(alpha, df1, df2)
    p_val = 2 * min(stats.f.cdf(f_val, df1, df2), stats.f.sf(f_val, df1, df2))
    return EffectSizeResult(
        measure="Variance ratio (F)",
        estimate=float(f_val),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=len(x) + len(y),
        extra={"p_value": float(p_val), "df1": df1, "df2": df2},
    )


# ===================================================================
# CONVERSION FUNCTIONS
# ===================================================================


def d_to_r(d: float, n1: int | None = None, n2: int | None = None) -> float:
    """Convert Cohen's *d* to Pearson *r*.

    :math:`r = d / \\sqrt{d^2 + a}` where :math:`a = (n_1 + n_2)^2 / (n_1 n_2)`
    or *a = 4* when sample sizes are unknown.

    Parameters
    ----------
    d : float
    n1, n2 : int or None

    Returns
    -------
    float
    """
    a = (n1 + n2) ** 2 / (n1 * n2) if n1 is not None and n2 is not None else 4.0
    return d / math.sqrt(d**2 + a)


def r_to_d(r: float) -> float:
    """Convert Pearson *r* to Cohen's *d*.

    :math:`d = 2r / \\sqrt{1 - r^2}`

    Parameters
    ----------
    r : float

    Returns
    -------
    float
    """
    return 2 * r / math.sqrt(1 - r**2) if abs(r) < 1 else np.inf * np.sign(r)


def or_to_d(or_val: float) -> float:
    """Convert odds ratio to Cohen's *d* (Hasselblad & Hedges, 1995).

    :math:`d = \\log(OR) \\cdot \\sqrt{3} / \\pi`

    Parameters
    ----------
    or_val : float
        Odds ratio.

    Returns
    -------
    float
    """
    return math.log(or_val) * math.sqrt(3) / math.pi if or_val > 0 else 0.0


def d_to_or(d: float) -> float:
    """Convert Cohen's *d* to an odds ratio.

    :math:`OR = \\exp(d \\pi / \\sqrt{3})`

    Parameters
    ----------
    d : float

    Returns
    -------
    float
    """
    return math.exp(d * math.pi / math.sqrt(3))


def or_to_r(or_val: float) -> float:
    """Convert odds ratio to Pearson *r* via *d*."""
    return d_to_r(or_to_d(or_val))


def r_to_or(r: float) -> float:
    """Convert Pearson *r* to odds ratio via *d*."""
    return d_to_or(r_to_d(r))


def d_to_nnt(d: float, base_rate: float = 0.5) -> float:
    """Convert Cohen's *d* to NNT given a base rate.

    Uses the Kraemer & Kupfer (2006) formula:
    :math:`NNT = 1 / (\\Phi(d/2 + \\Phi^{-1}(CER)) - CER)`

    Parameters
    ----------
    d : float
    base_rate : float, default 0.5
        Control event rate.

    Returns
    -------
    float
    """
    z_cer = stats.norm.ppf(base_rate)
    p_treat = stats.norm.cdf(d + z_cer)  # Note: approximate
    rd = p_treat - base_rate
    return 1 / abs(rd) if abs(rd) > 0 else np.inf


# ===================================================================
# META-ANALYSIS
# ===================================================================


def fixed_effects_meta(
    estimates: Union[np.ndarray, list[float]],
    standard_errors: Union[np.ndarray, list[float]],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Fixed-effects (inverse-variance weighted) meta-analytic pooling.

    Parameters
    ----------
    estimates : array-like
        Effect-size estimates from *k* studies.
    standard_errors : array-like
        Standard errors of the estimates.
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    theta = np.asarray(estimates, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    w = 1 / se**2
    pooled = (w * theta).sum() / w.sum()
    pooled_se = math.sqrt(1 / w.sum())
    z = stats.norm.ppf((1 + confidence) / 2)
    # Q statistic
    q = float(((theta - pooled) ** 2 * w).sum())
    k = len(theta)
    p_q = 1 - stats.chi2.cdf(q, k - 1) if k > 1 else 1.0
    return EffectSizeResult(
        measure="Fixed-effects meta-analysis",
        estimate=float(pooled),
        ci_lower=float(pooled - z * pooled_se),
        ci_upper=float(pooled + z * pooled_se),
        se=float(pooled_se),
        n=k,
        extra={"Q": float(q), "Q_p_value": float(p_q)},
    )


def random_effects_meta(
    estimates: Union[np.ndarray, list[float]],
    standard_errors: Union[np.ndarray, list[float]],
    confidence: float = 0.95,
    method: str = "DL",
) -> EffectSizeResult:
    """Random-effects meta-analytic pooling.

    Parameters
    ----------
    estimates : array-like
    standard_errors : array-like
    confidence : float, default 0.95
    method : str, default "DL"
        Tau-squared estimator: ``"DL"`` (DerSimonian--Laird), ``"PM"``
        (Paule--Mandel) or ``"REML"``.

    Returns
    -------
    EffectSizeResult

    References
    ----------
    DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
    *Controlled Clinical Trials*, 7(3), 177--188.
    """
    theta = np.asarray(estimates, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    k = len(theta)
    w = 1 / se**2

    # Fixed-effects pooled
    theta_fe = (w * theta).sum() / w.sum()
    Q = float(((theta - theta_fe) ** 2 * w).sum())

    # DerSimonian-Laird tau-squared
    c = w.sum() - (w**2).sum() / w.sum()
    if method == "DL":
        tau2 = max((Q - (k - 1)) / c, 0.0) if c > 0 else 0.0
    elif method == "PM":
        # Paule-Mandel: the root of the generalised Q statistic,
        # Q(tau2) = k - 1 (Q decreases in tau2), by bisection
        def qg(t):
            wt = 1 / (se**2 + t)
            return float((wt * (theta - (wt * theta).sum() / wt.sum()) ** 2).sum())

        if k < 2 or qg(0.0) <= k - 1:
            tau2 = 0.0
        else:
            lo, hi = 0.0, max((Q - (k - 1)) / c if c > 0 else 0.0, 1e-8)
            while qg(hi) > k - 1:
                hi *= 2
            for _ in range(400):
                mid = (lo + hi) / 2
                if qg(mid) > k - 1:
                    lo = mid
                else:
                    hi = mid
                if hi - lo <= 4 * np.finfo(float).eps * hi:
                    break
            tau2 = (lo + hi) / 2
    elif method == "REML":
        # REML by the fixed-point iteration of Viechtbauer (2005), eq. 11,
        # from the DL start
        tau2 = max((Q - (k - 1)) / c, 0.0) if c > 0 else 0.0
        for _ in range(10000):
            wt = 1 / (se**2 + tau2)
            mu = (wt * theta).sum() / wt.sum()
            new = max(
                (float((wt**2 * ((theta - mu) ** 2 - se**2)).sum()) + float((wt**2).sum()) / float(wt.sum()))
                / float((wt**2).sum()),
                0.0,
            )
            done = abs(new - tau2) <= 1e-15 * max(1.0, tau2)
            tau2 = new
            if done:
                break
    else:
        raise ValueError(f"method must be 'DL', 'PM' or 'REML' (got {method!r})")

    # Random-effects weights
    w_re = 1 / (se**2 + tau2)
    pooled = (w_re * theta).sum() / w_re.sum()
    pooled_se = math.sqrt(1 / w_re.sum())
    z = stats.norm.ppf((1 + confidence) / 2)

    # I-squared
    i2 = max((Q - (k - 1)) / Q, 0.0) * 100 if Q > 0 else 0.0

    # Prediction interval
    pred_se = math.sqrt(pooled_se**2 + tau2)
    t_crit = stats.t.ppf((1 + confidence) / 2, max(k - 2, 1))
    pred_lo = pooled - t_crit * pred_se
    pred_hi = pooled + t_crit * pred_se

    return EffectSizeResult(
        measure=f"Random-effects meta-analysis ({method})",
        estimate=float(pooled),
        ci_lower=float(pooled - z * pooled_se),
        ci_upper=float(pooled + z * pooled_se),
        se=float(pooled_se),
        n=k,
        extra={
            "tau_squared": float(tau2),
            "tau": float(math.sqrt(tau2)),
            "I_squared": float(i2),
            "Q": float(Q),
            "Q_p_value": float(stats.chi2.sf(Q, k - 1)) if k > 1 else 1.0,
            "prediction_interval_lower": float(pred_lo),
            "prediction_interval_upper": float(pred_hi),
        },
    )


def i_squared(
    estimates: Union[np.ndarray, list[float]],
    standard_errors: Union[np.ndarray, list[float]],
) -> float:
    """Compute Higgins' *I*-squared heterogeneity statistic.

    :math:`I^2 = \\max(0, (Q - (k-1))/Q) \\times 100`

    Parameters
    ----------
    estimates : array-like
    standard_errors : array-like

    Returns
    -------
    float
        Percentage (0--100).
    """
    result = random_effects_meta(estimates, standard_errors)
    return result.extra.get("I_squared", 0.0)


def prediction_interval(
    estimates: Union[np.ndarray, list[float]],
    standard_errors: Union[np.ndarray, list[float]],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Prediction interval for a new study from a random-effects meta-analysis.

    Parameters
    ----------
    estimates : array-like
    standard_errors : array-like
    confidence : float, default 0.95

    Returns
    -------
    tuple[float, float]
        (lower, upper) prediction interval bounds.
    """
    result = random_effects_meta(estimates, standard_errors, confidence=confidence)
    return (
        result.extra["prediction_interval_lower"],
        result.extra["prediction_interval_upper"],
    )


# ===================================================================
# BOOTSTRAP CI FOR ARBITRARY EFFECT SIZE
# ===================================================================


def bootstrap_effect_size_ci(
    func: callable,
    *arrays: Union[np.ndarray, pd.Series, list],
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> EffectSizeResult:
    """Generic bootstrap CI wrapper for any effect-size function.

    Parameters
    ----------
    func : callable
        Function that takes one or more arrays and returns a scalar.
    *arrays : array-like
        Input arrays to bootstrap.
    n_boot : int, default 2000
    confidence : float, default 0.95
    seed : int, default 42

    Returns
    -------
    EffectSizeResult
    """
    arrs = tuple(_arr(a) for a in arrays)
    point = float(func(*arrs))
    se, ci_lo, ci_hi = _bootstrap_ci(func, arrs, n_boot=n_boot, confidence=confidence, seed=seed)
    return EffectSizeResult(
        measure=f"Bootstrap ({func.__name__})",
        estimate=point,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=sum(len(a) for a in arrs),
    )
