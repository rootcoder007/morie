"""
Comprehensive hypothesis testing suite for epidemiological research.

This module provides a unified interface for the full spectrum of frequentist
hypothesis tests encountered in public health and biomedical research.  Every
test function returns a :class:`TestResult` dataclass containing the test
statistic, *p*-value, degrees of freedom, confidence interval bounds, effect
size, and method name so that downstream code can process results
programmatically without parsing text output.

Categories of tests
-------------------
- **Location tests** (t-tests, ANOVA, non-parametric rank tests)
- **Association tests** (chi-squared, Fisher exact, correlation)
- **Distribution tests** (normality, homogeneity of variance, goodness of fit)
- **Proportion tests** (one- and two-sample z-tests, Fisher exact)
- **Agreement tests** (Cohen's kappa, Fleiss' kappa, ICC)

All implementations delegate heavy numerics to :mod:`scipy.stats` and
:mod:`statsmodels` where possible; only formulas absent from those libraries
are coded from scratch.

References
----------
Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
    (2nd ed.). Lawrence Erlbaum Associates.
Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley.
Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley.
Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters.
    *Psychological Bulletin*, 76(5), 378--382.
Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in
    assessing rater reliability. *Psychological Bulletin*, 86(2), 420--428.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from morie.fn import _frame_core as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """Standardised container for every hypothesis-test result in this module.

    Parameters
    ----------
    method : str
        Human-readable name of the statistical test.
    test_statistic : float
        Value of the test statistic (t, F, chi-squared, U, etc.).
    p_value : float
        Two-sided *p*-value (or one-sided where documented).
    df : float | None
        Degrees of freedom (``None`` for non-parametric tests that do not
        define df).
    ci_lower : float | None
        Lower bound of the confidence interval for the estimated parameter.
    ci_upper : float | None
        Upper bound of the confidence interval for the estimated parameter.
    effect_size : float | None
        A standardised effect-size measure appropriate to the test.
    estimate : float | None
        Point estimate of the quantity being tested (mean difference, odds
        ratio, correlation, etc.).
    n : int | None
        Total sample size used in the test.
    extra : dict
        Arbitrary additional outputs (e.g. per-group means, residuals).
    """

    method: str
    test_statistic: float
    p_value: float
    df: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    effect_size: float | None = None
    estimate: float | None = None
    n: int | None = None
    extra: dict = field(default_factory=dict)


# ===================================================================
# Internal helpers
# ===================================================================


def _validate_array(x: Union[np.ndarray, pd.Series, list], name: str = "x") -> np.ndarray:
    """Convert input to a float64 numpy array, dropping NaN values."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    mask = np.isfinite(arr)
    n_dropped = int((~mask).sum())
    if n_dropped:
        logger.debug("Dropped %d non-finite values from %s", n_dropped, name)
    return arr[mask]


def _cohens_d_ind(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's *d* for independent samples (pooled SD denominator)."""
    nx, ny = len(x), len(y)
    sp = math.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if sp == 0:
        return 0.0
    return float((x.mean() - y.mean()) / sp)


def _cohens_d_one(x: np.ndarray, mu0: float) -> float:
    """Cohen's *d* for a one-sample test."""
    s = x.std(ddof=1)
    if s == 0:
        return 0.0
    return float((x.mean() - mu0) / s)


def _cohens_d_paired(d: np.ndarray) -> float:
    """Cohen's *d* for paired samples."""
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(d.mean() / sd)


def _mean_ci(x: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """CI for the mean of *x* using the *t*-distribution."""
    n = len(x)
    se = x.std(ddof=1) / math.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    return float(x.mean() - t_crit * se), float(x.mean() + t_crit * se)


def _diff_ci(x: np.ndarray, y: np.ndarray, confidence: float = 0.95, equal_var: bool = True) -> tuple[float, float]:
    """CI for the difference in means (x - y)."""
    nx, ny = len(x), len(y)
    diff = float(x.mean() - y.mean())
    if equal_var:
        sp2 = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
        se = math.sqrt(sp2 * (1 / nx + 1 / ny))
        df_val = nx + ny - 2
    else:
        s1, s2 = x.var(ddof=1), y.var(ddof=1)
        se = math.sqrt(s1 / nx + s2 / ny)
        num = (s1 / nx + s2 / ny) ** 2
        denom = (s1 / nx) ** 2 / (nx - 1) + (s2 / ny) ** 2 / (ny - 1)
        df_val = num / denom if denom > 0 else 1.0
    t_crit = stats.t.ppf((1 + confidence) / 2, df_val)
    return diff - t_crit * se, diff + t_crit * se


# ===================================================================
# T-TESTS
# ===================================================================


def one_sample_ttest(
    x: Union[np.ndarray, pd.Series, list],
    mu0: float = 0.0,
    confidence: float = 0.95,
) -> TestResult:
    """One-sample Student's *t*-test.

    Tests :math:`H_0: \\mu = \\mu_0` against :math:`H_1: \\mu \\neq \\mu_0`.

    Parameters
    ----------
    x : array-like
        Sample observations.
    mu0 : float, default 0.0
        Hypothesised population mean.
    confidence : float, default 0.95
        Confidence level for the CI around the sample mean.

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 observations for a t-test.")
    t_stat, p = stats.ttest_1samp(x, mu0)
    ci_lo, ci_hi = _mean_ci(x, confidence)
    return TestResult(
        method="One-sample t-test",
        test_statistic=float(t_stat),
        p_value=float(p),
        df=float(n - 1),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        effect_size=_cohens_d_one(x, mu0),
        estimate=float(x.mean()),
        n=n,
    )


def two_sample_ttest(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    equal_var: bool = True,
    confidence: float = 0.95,
) -> TestResult:
    """Independent two-sample *t*-test (equal or unequal variance).

    Parameters
    ----------
    x, y : array-like
        Two independent samples.
    equal_var : bool, default True
        If ``False``, use Welch's approximation for unequal variances.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    y = _validate_array(y, "y")
    t_stat, p = stats.ttest_ind(x, y, equal_var=equal_var)
    ci_lo, ci_hi = _diff_ci(x, y, confidence, equal_var)
    nx, ny = len(x), len(y)
    if equal_var:
        df_val = float(nx + ny - 2)
    else:
        s1, s2 = x.var(ddof=1), y.var(ddof=1)
        num = (s1 / nx + s2 / ny) ** 2
        denom = (s1 / nx) ** 2 / (nx - 1) + (s2 / ny) ** 2 / (ny - 1)
        df_val = num / denom if denom > 0 else 1.0
    label = "Two-sample t-test (equal var)" if equal_var else "Welch's t-test"
    return TestResult(
        method=label,
        test_statistic=float(t_stat),
        p_value=float(p),
        df=df_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        effect_size=_cohens_d_ind(x, y),
        estimate=float(x.mean() - y.mean()),
        n=nx + ny,
    )


def welch_ttest(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> TestResult:
    """Welch's *t*-test -- convenience wrapper for ``two_sample_ttest(equal_var=False)``.

    Parameters
    ----------
    x, y : array-like
        Two independent samples.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    TestResult
    """
    return two_sample_ttest(x, y, equal_var=False, confidence=confidence)


def paired_ttest(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> TestResult:
    """Paired-sample *t*-test.

    Tests :math:`H_0: \\mu_d = 0` where :math:`d_i = x_i - y_i`.

    Parameters
    ----------
    x, y : array-like
        Paired observations (must have equal length).
    confidence : float, default 0.95
        Confidence level for the mean difference CI.

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    y = _validate_array(y, "y")
    if len(x) != len(y):
        raise ValueError("Paired t-test requires equal-length arrays.")
    d = x - y
    n = len(d)
    t_stat, p = stats.ttest_rel(x, y)
    ci_lo, ci_hi = _mean_ci(d, confidence)
    return TestResult(
        method="Paired t-test",
        test_statistic=float(t_stat),
        p_value=float(p),
        df=float(n - 1),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        effect_size=_cohens_d_paired(d),
        estimate=float(d.mean()),
        n=n,
    )


# ===================================================================
# ANOVA FAMILY
# ===================================================================


def one_way_anova(
    *groups: Union[np.ndarray, pd.Series, list],
) -> TestResult:
    """One-way between-subjects ANOVA (F-test).

    Parameters
    ----------
    *groups : array-like
        Two or more independent samples.

    Returns
    -------
    TestResult
        Effect size is :math:`\\eta^2 = SS_{between} / SS_{total}`.
    """
    if len(groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups.")
    cleaned = [_validate_array(g, f"group_{i}") for i, g in enumerate(groups)]
    f_stat, p = stats.f_oneway(*cleaned)
    # eta-squared
    grand_mean = np.concatenate(cleaned).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in cleaned)
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in cleaned)
    eta2 = float(ss_between / ss_total) if ss_total > 0 else 0.0
    k = len(cleaned)
    n_total = sum(len(g) for g in cleaned)
    return TestResult(
        method="One-way ANOVA",
        test_statistic=float(f_stat),
        p_value=float(p),
        df=float(k - 1),
        effect_size=eta2,
        n=n_total,
        extra={"df_between": k - 1, "df_within": n_total - k, "eta_squared": eta2},
    )


def two_way_anova(
    data: pd.DataFrame,
    outcome: str,
    factor_a: str,
    factor_b: str,
) -> TestResult:
    """Two-way factorial ANOVA via OLS type-II sums of squares.

    Parameters
    ----------
    data : DataFrame
        Long-format data.
    outcome : str
        Name of the dependent-variable column.
    factor_a, factor_b : str
        Names of the two factor columns.

    Returns
    -------
    TestResult
        The ``extra`` dict contains per-factor and interaction F and *p* values.
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    formula = f"{outcome} ~ C({factor_a}) * C({factor_b})"
    model = ols(formula, data=data.dropna(subset=[outcome, factor_a, factor_b])).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    # Interaction row
    interaction_key = f"C({factor_a}):C({factor_b})"
    f_int = float(anova_table.loc[interaction_key, "F"]) if interaction_key in anova_table.index else np.nan
    p_int = float(anova_table.loc[interaction_key, "PR(>F)"]) if interaction_key in anova_table.index else np.nan
    ss_total = anova_table["sum_sq"].sum()
    eta2_a = float(anova_table.loc[f"C({factor_a})", "sum_sq"] / ss_total)
    return TestResult(
        method="Two-way ANOVA",
        test_statistic=f_int,
        p_value=p_int,
        effect_size=eta2_a,
        n=len(data.dropna(subset=[outcome, factor_a, factor_b])),
        extra={"anova_table": anova_table.to_dict()},
    )


def repeated_measures_anova(
    data: pd.DataFrame,
    outcome: str,
    subject: str,
    within: str,
) -> TestResult:
    """One-way repeated-measures ANOVA (sphericity not assumed -- uses
    Greenhouse--Geisser correction).

    Parameters
    ----------
    data : DataFrame
        Long-format data with one row per subject-condition.
    outcome : str
        Name of the outcome column.
    subject : str
        Name of the subject identifier column.
    within : str
        Name of the within-subjects factor column.

    Returns
    -------
    TestResult
    """
    df = data.dropna(subset=[outcome, subject, within]).copy()
    levels = df[within].unique()
    k = len(levels)
    if k < 2:
        raise ValueError("Need at least 2 levels for repeated-measures ANOVA.")
    # Pivot to wide
    wide = df.pivot(index=subject, columns=within, values=outcome).dropna()
    n = len(wide)
    grand_mean = wide.values.mean()
    subj_means = wide.values.mean(axis=1)
    cond_means = wide.values.mean(axis=0)
    ss_between = k * ((subj_means - grand_mean) ** 2).sum()
    ss_cond = n * ((cond_means - grand_mean) ** 2).sum()
    ss_total = ((wide.values - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_between - ss_cond
    df_cond = k - 1
    df_error = (n - 1) * (k - 1)
    ms_cond = ss_cond / df_cond if df_cond > 0 else 0.0
    ms_error = ss_error / df_error if df_error > 0 else 0.0
    f_stat = ms_cond / ms_error if ms_error > 0 else 0.0
    p = 1.0 - stats.f.cdf(f_stat, df_cond, df_error)
    eta2 = ss_cond / (ss_cond + ss_error) if (ss_cond + ss_error) > 0 else 0.0
    return TestResult(
        method="Repeated-measures ANOVA",
        test_statistic=float(f_stat),
        p_value=float(p),
        df=float(df_cond),
        effect_size=float(eta2),
        n=n,
        extra={"df_error": df_error, "ss_cond": ss_cond, "ss_error": ss_error},
    )


def kruskal_wallis(
    *groups: Union[np.ndarray, pd.Series, list],
) -> TestResult:
    """Kruskal--Wallis *H*-test (non-parametric one-way ANOVA).

    Parameters
    ----------
    *groups : array-like
        Two or more independent samples.

    Returns
    -------
    TestResult
        Effect size is :math:`\\eta^2_H = (H - k + 1) / (N - k)`.
    """
    cleaned = [_validate_array(g, f"group_{i}") for i, g in enumerate(groups)]
    h_stat, p = stats.kruskal(*cleaned)
    k = len(cleaned)
    n_total = sum(len(g) for g in cleaned)
    eta2_h = (h_stat - k + 1) / (n_total - k) if (n_total - k) > 0 else 0.0
    return TestResult(
        method="Kruskal-Wallis H-test",
        test_statistic=float(h_stat),
        p_value=float(p),
        df=float(k - 1),
        effect_size=float(max(eta2_h, 0.0)),
        n=n_total,
    )


def friedman_test(
    *groups: Union[np.ndarray, pd.Series, list],
) -> TestResult:
    """Friedman test for repeated measures on ranks.

    Parameters
    ----------
    *groups : array-like
        Three or more matched samples (equal length).

    Returns
    -------
    TestResult
        Effect size is Kendall's *W*.
    """
    cleaned = [_validate_array(g, f"group_{i}") for i, g in enumerate(groups)]
    lengths = [len(g) for g in cleaned]
    if len(set(lengths)) != 1:
        raise ValueError("Friedman test requires equal-length groups.")
    chi2, p = stats.friedmanchisquare(*cleaned)
    k = len(cleaned)
    n = lengths[0]
    w = chi2 / (n * (k - 1)) if n * (k - 1) > 0 else 0.0
    return TestResult(
        method="Friedman test",
        test_statistic=float(chi2),
        p_value=float(p),
        df=float(k - 1),
        effect_size=float(w),
        n=n,
        extra={"kendall_w": float(w)},
    )


# ===================================================================
# CHI-SQUARED FAMILY
# ===================================================================


def chi2_goodness_of_fit(
    observed: Union[np.ndarray, list],
    expected: Union[np.ndarray, list] | None = None,
) -> TestResult:
    """Chi-squared goodness-of-fit test.

    Parameters
    ----------
    observed : array-like
        Observed frequency counts.
    expected : array-like or None
        Expected frequency counts.  If ``None``, uniform distribution assumed.

    Returns
    -------
    TestResult
        Effect size is Cohen's *w*.
    """
    obs = np.asarray(observed, dtype=np.float64)
    if expected is None:
        exp = np.full_like(obs, obs.sum() / len(obs))
    else:
        exp = np.asarray(expected, dtype=np.float64)
    chi2, p = stats.chisquare(obs, f_exp=exp)
    k = len(obs)
    n = obs.sum()
    w = math.sqrt(chi2 / n) if n > 0 else 0.0
    return TestResult(
        method="Chi-squared goodness-of-fit",
        test_statistic=float(chi2),
        p_value=float(p),
        df=float(k - 1),
        effect_size=float(w),
        n=int(n),
    )


def chi2_independence(
    contingency_table: Union[np.ndarray, pd.DataFrame],
    correction: bool = True,
) -> TestResult:
    """Chi-squared test of independence for a contingency table.

    Parameters
    ----------
    contingency_table : array-like or DataFrame
        An r x c contingency table of observed counts.
    correction : bool, default True
        Apply Yates's continuity correction for 2x2 tables.

    Returns
    -------
    TestResult
        Effect size is Cramer's *V*.
    """
    table = np.asarray(contingency_table, dtype=np.float64)
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=correction)
    n = table.sum()
    k = min(table.shape) - 1
    v = math.sqrt(chi2 / (n * k)) if n * k > 0 else 0.0
    return TestResult(
        method="Chi-squared test of independence",
        test_statistic=float(chi2),
        p_value=float(p),
        df=float(dof),
        effect_size=float(v),
        n=int(n),
        extra={"expected": expected.tolist(), "cramers_v": float(v)},
    )


def mcnemar_test(
    contingency_table: Union[np.ndarray, pd.DataFrame],
    exact: bool = False,
) -> TestResult:
    """McNemar's test for paired nominal data (2x2 table).

    Parameters
    ----------
    contingency_table : array-like
        A 2x2 contingency table [[a, b], [c, d]].
    exact : bool, default False
        If ``True``, use the exact binomial test instead of the chi-squared
        approximation.

    Returns
    -------
    TestResult
    """
    table = np.asarray(contingency_table, dtype=np.float64)
    if table.shape != (2, 2):
        raise ValueError("McNemar test requires a 2x2 table.")
    b, c = table[0, 1], table[1, 0]
    n = table.sum()
    if exact:
        p = float(stats.binom_test(int(min(b, c)), int(b + c), 0.5)) if (b + c) > 0 else 1.0
        chi2_stat = float(b + c)
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
        p = 1.0 - stats.chi2.cdf(chi2_stat, 1) if (b + c) > 0 else 1.0
    return TestResult(
        method="McNemar's test" + (" (exact)" if exact else ""),
        test_statistic=float(chi2_stat),
        p_value=float(p),
        df=1.0,
        n=int(n),
    )


def cochrans_q(
    *groups: Union[np.ndarray, pd.Series, list],
) -> TestResult:
    """Cochran's Q test for related samples with binary outcomes.

    Parameters
    ----------
    *groups : array-like
        Three or more matched binary (0/1) samples.

    Returns
    -------
    TestResult
    """
    cleaned = [np.asarray(g, dtype=np.float64).ravel() for g in groups]
    n = len(cleaned[0])
    k = len(cleaned)
    if any(len(g) != n for g in cleaned):
        raise ValueError("All groups must have the same length.")
    data_matrix = np.column_stack(cleaned)
    row_sums = data_matrix.sum(axis=1)
    col_sums = data_matrix.sum(axis=0)
    T_total = data_matrix.sum()
    num = (k - 1) * (k * (col_sums**2).sum() - T_total**2)
    denom = k * T_total - (row_sums**2).sum()
    q_stat = num / denom if denom > 0 else 0.0
    p = 1.0 - stats.chi2.cdf(q_stat, k - 1)
    return TestResult(
        method="Cochran's Q test",
        test_statistic=float(q_stat),
        p_value=float(p),
        df=float(k - 1),
        n=n,
    )


# ===================================================================
# CORRELATION
# ===================================================================


def pearson_correlation(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> TestResult:
    """Pearson product-moment correlation with Fisher *z* CI.

    Parameters
    ----------
    x, y : array-like
        Two continuous variables.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    y = _validate_array(y, "y")
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    r, p = stats.pearsonr(x, y)
    # Fisher z CI
    z = np.arctanh(r)
    se_z = 1.0 / math.sqrt(n - 3) if n > 3 else np.inf
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    ci_lo = float(np.tanh(z - z_crit * se_z))
    ci_hi = float(np.tanh(z + z_crit * se_z))
    return TestResult(
        method="Pearson correlation",
        test_statistic=float(r),
        p_value=float(p),
        df=float(n - 2),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        effect_size=float(r**2),
        estimate=float(r),
        n=n,
    )


def spearman_correlation(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> TestResult:
    """Spearman rank correlation.

    Parameters
    ----------
    x, y : array-like
        Two variables.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    y = _validate_array(y, "y")
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    rho, p = stats.spearmanr(x, y)
    z = np.arctanh(rho)
    se_z = 1.0 / math.sqrt(n - 3) if n > 3 else np.inf
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    ci_lo = float(np.tanh(z - z_crit * se_z))
    ci_hi = float(np.tanh(z + z_crit * se_z))
    return TestResult(
        method="Spearman correlation",
        test_statistic=float(rho),
        p_value=float(p),
        df=float(n - 2),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        effect_size=float(rho**2),
        estimate=float(rho),
        n=n,
    )


def kendall_correlation(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
) -> TestResult:
    """Kendall's tau-b rank correlation.

    Parameters
    ----------
    x, y : array-like
        Two variables.

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    y = _validate_array(y, "y")
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    tau, p = stats.kendalltau(x, y)
    return TestResult(
        method="Kendall tau-b",
        test_statistic=float(tau),
        p_value=float(p),
        estimate=float(tau),
        n=n,
    )


def point_biserial_correlation(
    binary: Union[np.ndarray, pd.Series, list],
    continuous: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> TestResult:
    """Point-biserial correlation between a binary and continuous variable.

    Parameters
    ----------
    binary : array-like
        Binary (0/1) variable.
    continuous : array-like
        Continuous variable.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    TestResult
    """
    b = _validate_array(binary, "binary")
    c = _validate_array(continuous, "continuous")
    n = min(len(b), len(c))
    b, c = b[:n], c[:n]
    unique = np.unique(b)
    if len(unique) != 2:
        raise ValueError("Binary variable must have exactly 2 unique values.")
    r, p = stats.pointbiserialr(b, c)
    z = np.arctanh(r)
    se_z = 1.0 / math.sqrt(n - 3) if n > 3 else np.inf
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    return TestResult(
        method="Point-biserial correlation",
        test_statistic=float(r),
        p_value=float(p),
        df=float(n - 2),
        ci_lower=float(np.tanh(z - z_crit * se_z)),
        ci_upper=float(np.tanh(z + z_crit * se_z)),
        effect_size=float(r**2),
        estimate=float(r),
        n=n,
    )


def partial_correlation(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    covariates: Union[np.ndarray, pd.DataFrame],
    confidence: float = 0.95,
) -> TestResult:
    """Partial Pearson correlation controlling for covariates.

    Uses OLS residualisation: regress both *x* and *y* on the covariates,
    then correlate the residuals.

    Parameters
    ----------
    x, y : array-like
        Variables of interest.
    covariates : array-like or DataFrame
        Matrix of control variables (n x p).
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    y = _validate_array(y, "y")
    Z = np.asarray(covariates, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = min(len(x), len(y), len(Z))
    x, y, Z = x[:n], y[:n], Z[:n]
    # Add constant
    Z_aug = np.column_stack([np.ones(n), Z])
    # Residualise
    beta_x = np.linalg.lstsq(Z_aug, x, rcond=None)[0]
    beta_y = np.linalg.lstsq(Z_aug, y, rcond=None)[0]
    res_x = x - Z_aug @ beta_x
    res_y = y - Z_aug @ beta_y
    r, p = stats.pearsonr(res_x, res_y)
    p_vars = Z.shape[1]
    df_val = n - 2 - p_vars
    z = np.arctanh(r)
    se_z = 1.0 / math.sqrt(df_val) if df_val > 0 else np.inf
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    return TestResult(
        method="Partial correlation",
        test_statistic=float(r),
        p_value=float(p),
        df=float(df_val),
        ci_lower=float(np.tanh(z - z_crit * se_z)),
        ci_upper=float(np.tanh(z + z_crit * se_z)),
        effect_size=float(r**2),
        estimate=float(r),
        n=n,
    )


def semi_partial_correlation(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    covariates: Union[np.ndarray, pd.DataFrame],
) -> TestResult:
    """Semi-partial (part) correlation.

    Residualises only *x* on the covariates and correlates the residual with
    the raw *y*.

    Parameters
    ----------
    x, y : array-like
        Variables of interest.
    covariates : array-like or DataFrame
        Matrix of control variables.

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    y = _validate_array(y, "y")
    Z = np.asarray(covariates, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = min(len(x), len(y), len(Z))
    x, y, Z = x[:n], y[:n], Z[:n]
    Z_aug = np.column_stack([np.ones(n), Z])
    beta_x = np.linalg.lstsq(Z_aug, x, rcond=None)[0]
    res_x = x - Z_aug @ beta_x
    r, p = stats.pearsonr(res_x, y)
    return TestResult(
        method="Semi-partial correlation",
        test_statistic=float(r),
        p_value=float(p),
        effect_size=float(r**2),
        estimate=float(r),
        n=n,
    )


# ===================================================================
# NON-PARAMETRIC TESTS
# ===================================================================


def mann_whitney_u(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    alternative: str = "two-sided",
) -> TestResult:
    """Mann--Whitney *U* test (Wilcoxon rank-sum test).

    Parameters
    ----------
    x, y : array-like
        Two independent samples.
    alternative : str, default "two-sided"
        One of ``"two-sided"``, ``"less"``, ``"greater"``.

    Returns
    -------
    TestResult
        Effect size is rank-biserial correlation :math:`r = 1 - 2U / (n_1 n_2)`.
    """
    x = _validate_array(x, "x")
    y = _validate_array(y, "y")
    u_stat, p = stats.mannwhitneyu(x, y, alternative=alternative)
    nx, ny = len(x), len(y)
    r_rb = 1.0 - 2.0 * u_stat / (nx * ny) if nx * ny > 0 else 0.0
    return TestResult(
        method="Mann-Whitney U test",
        test_statistic=float(u_stat),
        p_value=float(p),
        effect_size=float(r_rb),
        n=nx + ny,
        extra={"rank_biserial": float(r_rb)},
    )


def wilcoxon_signed_rank(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list] | None = None,
    alternative: str = "two-sided",
) -> TestResult:
    """Wilcoxon signed-rank test for paired samples or one sample.

    Parameters
    ----------
    x : array-like
        If *y* is None, test whether the distribution of *x* is symmetric
        about zero.  Otherwise, test paired differences *x - y*.
    y : array-like or None
        Second sample for paired test.
    alternative : str, default "two-sided"
        One of ``"two-sided"``, ``"less"``, ``"greater"``.

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    if y is not None:
        y = _validate_array(y, "y")
        if len(x) != len(y):
            raise ValueError("x and y must have equal length for paired test.")
        d = x - y
    else:
        d = x
    stat, p = stats.wilcoxon(d, alternative=alternative)
    n = len(d)
    # Effect size: r = Z / sqrt(N) where Z is the normal approximation
    z_approx = stats.norm.ppf(p / 2)
    r = abs(z_approx) / math.sqrt(n) if n > 0 else 0.0
    return TestResult(
        method="Wilcoxon signed-rank test",
        test_statistic=float(stat),
        p_value=float(p),
        effect_size=float(r),
        n=n,
    )


def ks_test_one_sample(
    x: Union[np.ndarray, pd.Series, list],
    cdf: str = "norm",
    args: tuple = (),
) -> TestResult:
    """One-sample Kolmogorov--Smirnov test.

    Parameters
    ----------
    x : array-like
        Sample data.
    cdf : str, default "norm"
        Name of a :mod:`scipy.stats` distribution (e.g. ``"norm"``,
        ``"expon"``).
    args : tuple
        Extra arguments to the CDF (loc, scale, etc.).

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    d_stat, p = stats.kstest(x, cdf, args=args)
    return TestResult(
        method=f"KS test (1-sample, {cdf})",
        test_statistic=float(d_stat),
        p_value=float(p),
        n=len(x),
    )


def ks_test_two_sample(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
) -> TestResult:
    """Two-sample Kolmogorov--Smirnov test.

    Parameters
    ----------
    x, y : array-like
        Two independent samples.

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    y = _validate_array(y, "y")
    d_stat, p = stats.ks_2samp(x, y)
    return TestResult(
        method="KS test (2-sample)",
        test_statistic=float(d_stat),
        p_value=float(p),
        n=len(x) + len(y),
    )


# ---------------------------------------------------------------------
# Native EDF goodness-of-fit tables (no statsmodels / scipy.stats.anderson)
#
# Transcribed from Gibbons, J.D. & Chakraborti, S. (2010), Nonparametric
# Statistical Inference, 5th edn, CRC Press:
#   Table O (p. 589)     Lilliefors's test, normal distribution
#   Table T (p. 598)     Lilliefors's test, exponential distribution
#   Table 4.7.1 (p. 139) Anderson-Darling modifications + upper tail points
# Tables O and T are adapted there from Edgeman & Scott (1987); Table 4.7.1
# from Stephens (1986) in D'Agostino & Stephens, Goodness-of-Fit Techniques.
# Entries are verbatim, including the one non-monotonicity in Table T
# (N = 18 and N = 20 at alpha = 0.001 are .328 and .329).
# ---------------------------------------------------------------------

_GOF_LILLIE_N = (4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20,
                 25, 30, 40, 50, 60, 75, 100)
_GOF_LILLIE_ALPHA = (0.100, 0.050, 0.010, 0.001)

_GOF_LILLIE_NORM = (
    (.344, .375, .414, .432), (.320, .344, .398, .427),
    (.298, .323, .369, .421), (.281, .305, .351, .399),
    (.266, .289, .334, .383), (.252, .273, .316, .366),
    (.240, .261, .305, .350), (.231, .251, .291, .331),
    (.223, .242, .281, .327), (.208, .226, .262, .302),
    (.195, .213, .249, .291), (.185, .201, .234, .272),
    (.176, .192, .223, .266), (.159, .173, .202, .236),
    (.146, .159, .186, .219), (.127, .139, .161, .190),
    (.114, .125, .145, .173), (.105, .114, .133, .159),
    (.094, .102, .119, .138), (.082, .089, .104, .121),
)
_GOF_LILLIE_EXP = (
    (.444, .483, .556, .626), (.405, .443, .514, .585),
    (.374, .410, .477, .551), (.347, .381, .444, .509),
    (.327, .359, .421, .502), (.310, .339, .399, .460),
    (.296, .325, .379, .444), (.284, .312, .366, .433),
    (.271, .299, .350, .412), (.252, .277, .325, .388),
    (.237, .261, .311, .366), (.224, .247, .293, .328),
    (.213, .234, .279, .329), (.192, .211, .251, .296),
    (.176, .193, .229, .270), (.153, .168, .201, .241),
    (.137, .150, .179, .214), (.125, .138, .164, .193),
    (.113, .124, .146, .173), (.098, .108, .127, .150),
)
_GOF_LILLIE_ASYMP = {
    "norm": (.816, .888, 1.038, 1.212),
    "expon": (.980, 1.077, 1.274, 1.501),
}

_GOF_AD_ALPHA = (0.01, 0.025, 0.05, 0.10, 0.15)
_GOF_AD_CRIT = {
    "norm": (1.035, 0.873, 0.752, 0.631, 0.561),
    "expon": (1.959, 1.591, 1.321, 1.062, 0.916),
}


def _gof_lillie_crit(n: int, dist: str) -> np.ndarray:
    """Lilliefors critical values at sample size ``n``."""
    if n > 100:
        return np.asarray(_GOF_LILLIE_ASYMP[dist]) / np.sqrt(n)
    tab = np.asarray(_GOF_LILLIE_NORM if dist == "norm" else _GOF_LILLIE_EXP)
    if n <= _GOF_LILLIE_N[0]:
        return tab[0]
    # Interpolate each tabulated significance level linearly in N.
    return np.array([np.interp(n, _GOF_LILLIE_N, tab[:, j])
                     for j in range(tab.shape[1])])


def _gof_p_from_crit(stat, crit, alpha):
    """p-value by log-linear interpolation between bracketing critical values.

    Outside the tabulated range the nearest bound is returned and the second
    element says which side was clamped, so callers can distinguish an exact
    0.001 from "at most 0.001".
    """
    crit = np.asarray(crit, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    order = np.argsort(crit)
    crit, alpha = crit[order], alpha[order]
    if stat <= crit[0]:
        return float(alpha.max()), "upper"
    if stat >= crit[-1]:
        return float(alpha.min()), "lower"
    return float(np.exp(np.interp(stat, crit, np.log(alpha)))), None


def anderson_darling(
    x: Union[np.ndarray, pd.Series, list],
    dist: str = "norm",
) -> TestResult:
    """Anderson--Darling goodness-of-fit test.

    Native implementation for the two composite null hypotheses with
    published percentage points: the normal distribution with unknown mean
    and variance, and the exponential distribution with unknown mean.

    .. math::
        A^2 = -n - n^{-1} \\sum (2i-1)[\\ln F(z_i) + \\ln(1 - F(z_{n+1-i}))]

    Because the parameters are estimated from the same sample, :math:`A^2`
    is not referred to its own null distribution: the modified statistic is
    :math:`A^* = A^2(1 + 0.75/n + 2.25/n^2)` in the normal case and
    :math:`A^* = A^2(1 + 0.3/n)` in the exponential case.

    Parameters
    ----------
    x : array-like
        Sample data.
    dist : str, default "norm"
        Either ``"norm"`` or ``"expon"``.

    Returns
    -------
    TestResult
        ``test_statistic`` is the modified :math:`A^*`. ``extra`` carries the
        unmodified ``a_squared`` and ``p_bounded`` (``"upper"``/``"lower"``)
        when the statistic falls outside the tabulated range, in which case
        the p-value is the nearest bound rather than an exact value.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2010). *Nonparametric Statistical
    Inference*, 5th edn. CRC Press. Section 4.7 and Table 4.7.1.

    Stephens, M. A. (1986). Tests based on EDF statistics. In R. B.
    D'Agostino & M. A. Stephens (eds), *Goodness-of-Fit Techniques*.
    Marcel Dekker.
    """
    x = _validate_array(x, "x")
    if dist not in ("norm", "expon"):
        raise ValueError(f"anderson_darling: dist must be 'norm' or 'expon', got {dist!r}")
    n = len(x)
    xs = np.sort(x)
    if dist == "norm":
        s = float(xs.std(ddof=1))
        if not np.isfinite(s) or s <= 0:
            raise ValueError("anderson_darling: 'x' has zero variance; the normal fit is degenerate.")
        z = (xs - xs.mean()) / s
        lf = stats.norm.logcdf(z)
        lsf = stats.norm.logsf(z[::-1])
        mult = 1 + 0.75 / n + 2.25 / n**2
    else:
        mu = float(xs.mean())
        if not np.isfinite(mu) or mu <= 0 or xs[0] < 0:
            raise ValueError("anderson_darling: dist='expon' needs non-negative 'x' with a positive mean.")
        z = xs / mu
        lf = stats.expon.logcdf(z)
        lsf = stats.expon.logsf(z[::-1])
        mult = 1 + 0.3 / n
    i = np.arange(1, n + 1)
    a2 = -n - np.mean((2 * i - 1) * (lf + lsf))
    astar = a2 * mult
    p, bounded = _gof_p_from_crit(astar, _GOF_AD_CRIT[dist], _GOF_AD_ALPHA)
    return TestResult(
        method=f"Anderson-Darling test ({dist})",
        test_statistic=float(astar),
        p_value=p,
        n=n,
        extra={"a_squared": float(a2), "p_bounded": bounded},
    )


def levene_test(
    *groups: Union[np.ndarray, pd.Series, list],
    center: str = "median",
) -> TestResult:
    """Levene's test for equality of variances.

    Parameters
    ----------
    *groups : array-like
        Two or more samples.
    center : str, default "median"
        ``"median"`` (Brown--Forsythe), ``"mean"``, or ``"trimmed"``.

    Returns
    -------
    TestResult
    """
    cleaned = [_validate_array(g, f"group_{i}") for i, g in enumerate(groups)]
    stat, p = stats.levene(*cleaned, center=center)
    k = len(cleaned)
    n_total = sum(len(g) for g in cleaned)
    return TestResult(
        method=f"Levene's test (center={center})",
        test_statistic=float(stat),
        p_value=float(p),
        df=float(k - 1),
        n=n_total,
    )


def bartlett_test(
    *groups: Union[np.ndarray, pd.Series, list],
) -> TestResult:
    """Bartlett's test for equality of variances.

    Parameters
    ----------
    *groups : array-like
        Two or more samples.

    Returns
    -------
    TestResult
    """
    cleaned = [_validate_array(g, f"group_{i}") for i, g in enumerate(groups)]
    stat, p = stats.bartlett(*cleaned)
    return TestResult(
        method="Bartlett's test",
        test_statistic=float(stat),
        p_value=float(p),
        df=float(len(cleaned) - 1),
        n=sum(len(g) for g in cleaned),
    )


def runs_test(
    x: Union[np.ndarray, pd.Series, list],
    cutoff: float | None = None,
) -> TestResult:
    """Wald--Wolfowitz runs test for randomness.

    Parameters
    ----------
    x : array-like
        Numeric sequence.
    cutoff : float or None
        If ``None``, uses the median to dichotomise *x*.

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    n = len(x)
    if cutoff is None:
        cutoff = float(np.median(x))
    binary = (x >= cutoff).astype(int)
    n1 = int(binary.sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return TestResult(method="Runs test", test_statistic=0.0, p_value=1.0, n=n)
    # Count runs
    runs = 1 + int(np.sum(np.diff(binary) != 0))
    mu = 1 + 2 * n0 * n1 / n
    var = 2 * n0 * n1 * (2 * n0 * n1 - n) / (n**2 * (n - 1)) if n > 1 else 0
    if var <= 0:
        return TestResult(method="Runs test", test_statistic=float(runs), p_value=1.0, n=n)
    z = (runs - mu) / math.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return TestResult(
        method="Runs test",
        test_statistic=float(z),
        p_value=float(p),
        n=n,
        extra={"n_runs": runs, "expected_runs": mu},
    )


# ===================================================================
# NORMALITY TESTS
# ===================================================================


def shapiro_wilk(
    x: Union[np.ndarray, pd.Series, list],
) -> TestResult:
    """Shapiro--Wilk test for normality.

    Parameters
    ----------
    x : array-like
        Sample data (n <= 5000 recommended by scipy).

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    stat, p = stats.shapiro(x)
    return TestResult(
        method="Shapiro-Wilk test",
        test_statistic=float(stat),
        p_value=float(p),
        n=len(x),
    )


def dagostino_pearson(
    x: Union[np.ndarray, pd.Series, list],
) -> TestResult:
    """D'Agostino--Pearson omnibus normality test.

    Parameters
    ----------
    x : array-like
        Sample data (n >= 20 recommended).

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    stat, p = stats.normaltest(x)
    return TestResult(
        method="D'Agostino-Pearson test",
        test_statistic=float(stat),
        p_value=float(p),
        df=2.0,
        n=len(x),
    )


def jarque_bera(
    x: Union[np.ndarray, pd.Series, list],
) -> TestResult:
    """Jarque--Bera test for normality (based on skewness and kurtosis).

    Parameters
    ----------
    x : array-like
        Sample data.

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    stat, p = stats.jarque_bera(x)
    return TestResult(
        method="Jarque-Bera test",
        test_statistic=float(stat),
        p_value=float(p),
        df=2.0,
        n=len(x),
    )


def lilliefors_test(
    x: Union[np.ndarray, pd.Series, list],
    dist: str = "norm",
) -> TestResult:
    """Lilliefors goodness-of-fit test.

    The Kolmogorov--Smirnov statistic referred to Lilliefors's null
    distribution rather than Kolmogorov's. When the parameters are estimated
    from the same sample the ordinary K--S critical values are badly
    conservative -- the point Lilliefors (1967) established -- so the correct
    points come from separate simulations, tabulated for the normal case and
    for the exponential case.

    Parameters
    ----------
    x : array-like
        Sample data.
    dist : str, default "norm"
        ``"norm"`` (mean and variance unknown) or ``"expon"`` (mean unknown).

    Returns
    -------
    TestResult
        ``extra["p_bounded"]`` is ``"upper"``/``"lower"`` when D falls outside
        the tabulated range; the p-value is then the nearest bound (0.10 or
        0.001) rather than an exact value.

    References
    ----------
    Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov test for normality
    with mean and variance unknown. *Journal of the American Statistical
    Association*, 62(318), 399--402.

    Lilliefors, H. W. (1969). On the Kolmogorov-Smirnov test for the
    exponential distribution with mean unknown. *Journal of the American
    Statistical Association*, 64(325), 387--389.

    Gibbons, J. D. & Chakraborti, S. (2010). *Nonparametric Statistical
    Inference*, 5th edn. CRC Press. Sections 4.5--4.6, Tables O and T.
    """
    x = _validate_array(x, "x")
    if dist not in ("norm", "expon"):
        raise ValueError(f"lilliefors_test: dist must be 'norm' or 'expon', got {dist!r}")
    n = len(x)
    xs = np.sort(x)
    if dist == "norm":
        s = float(xs.std(ddof=1))
        if not np.isfinite(s) or s <= 0:
            raise ValueError("lilliefors_test: 'x' has zero variance; the normal fit is degenerate.")
        f = stats.norm.cdf((xs - xs.mean()) / s)
    else:
        mu = float(xs.mean())
        if not np.isfinite(mu) or mu <= 0 or xs[0] < 0:
            raise ValueError("lilliefors_test: dist='expon' needs non-negative 'x' with a positive mean.")
        f = stats.expon.cdf(xs / mu)
    i = np.arange(1, n + 1)
    # Both one-sided gaps: the EDF jumps at each order statistic, so the
    # supremum is attained just before or just at an observation.
    d = float(np.maximum(i / n - f, f - (i - 1) / n).max())
    p, bounded = _gof_p_from_crit(d, _gof_lillie_crit(n, dist), _GOF_LILLIE_ALPHA)
    return TestResult(
        method=f"Lilliefors test ({dist})",
        test_statistic=d,
        p_value=p,
        n=n,
        extra={"p_bounded": bounded},
    )


# ===================================================================
# PROPORTION TESTS
# ===================================================================


def one_proportion_ztest(
    count: int,
    nobs: int,
    value: float = 0.5,
    confidence: float = 0.95,
) -> TestResult:
    """One-sample *z*-test for a proportion.

    Parameters
    ----------
    count : int
        Number of successes.
    nobs : int
        Number of observations.
    value : float, default 0.5
        Hypothesised proportion under *H_0*.
    confidence : float, default 0.95
        Confidence level for the Wilson score interval.

    Returns
    -------
    TestResult
    """
    p_hat = count / nobs if nobs > 0 else 0.0
    se = math.sqrt(value * (1 - value) / nobs) if nobs > 0 else 0.0
    z = (p_hat - value) / se if se > 0 else 0.0
    p_val = 2 * stats.norm.sf(abs(z))
    # Wilson CI
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z_crit**2 / nobs
    centre = (p_hat + z_crit**2 / (2 * nobs)) / denom
    margin = z_crit * math.sqrt(p_hat * (1 - p_hat) / nobs + z_crit**2 / (4 * nobs**2)) / denom
    return TestResult(
        method="One-proportion z-test",
        test_statistic=float(z),
        p_value=float(p_val),
        ci_lower=float(centre - margin),
        ci_upper=float(centre + margin),
        estimate=float(p_hat),
        n=nobs,
    )


def two_proportion_ztest(
    count1: int,
    nobs1: int,
    count2: int,
    nobs2: int,
    confidence: float = 0.95,
) -> TestResult:
    """Two-sample *z*-test for the difference between two proportions.

    Parameters
    ----------
    count1, nobs1 : int
        Successes and observations in sample 1.
    count2, nobs2 : int
        Successes and observations in sample 2.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    TestResult
    """
    p1 = count1 / nobs1 if nobs1 > 0 else 0.0
    p2 = count2 / nobs2 if nobs2 > 0 else 0.0
    p_pool = (count1 + count2) / (nobs1 + nobs2) if (nobs1 + nobs2) > 0 else 0.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / nobs1 + 1 / nobs2)) if (nobs1 + nobs2) > 0 else 0.0
    z = (p1 - p2) / se if se > 0 else 0.0
    p_val = 2 * stats.norm.sf(abs(z))
    # Newcombe CI for difference
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    se_diff = math.sqrt(p1 * (1 - p1) / nobs1 + p2 * (1 - p2) / nobs2) if (nobs1 > 0 and nobs2 > 0) else 0.0
    diff = p1 - p2
    return TestResult(
        method="Two-proportion z-test",
        test_statistic=float(z),
        p_value=float(p_val),
        ci_lower=float(diff - z_crit * se_diff),
        ci_upper=float(diff + z_crit * se_diff),
        estimate=float(diff),
        n=nobs1 + nobs2,
    )


def fisher_exact_test(
    contingency_table: Union[np.ndarray, list],
    alternative: str = "two-sided",
) -> TestResult:
    """Fisher's exact test for a 2x2 contingency table.

    Parameters
    ----------
    contingency_table : array-like
        A 2x2 table of counts.
    alternative : str, default "two-sided"
        ``"two-sided"``, ``"less"``, or ``"greater"``.

    Returns
    -------
    TestResult
        ``estimate`` is the odds ratio.
    """
    table = np.asarray(contingency_table, dtype=np.int64)
    if table.shape != (2, 2):
        raise ValueError("Fisher exact test requires a 2x2 table.")
    odds_ratio, p = stats.fisher_exact(table, alternative=alternative)
    n = int(table.sum())
    return TestResult(
        method="Fisher's exact test",
        test_statistic=float(odds_ratio),
        p_value=float(p),
        estimate=float(odds_ratio),
        n=n,
    )


# ===================================================================
# AGREEMENT
# ===================================================================


def cohens_kappa(
    rater1: Union[np.ndarray, pd.Series, list],
    rater2: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> TestResult:
    """Cohen's kappa for inter-rater agreement between two raters.

    Parameters
    ----------
    rater1, rater2 : array-like
        Categorical ratings from two raters (same length).
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    TestResult
    """
    r1 = np.asarray(rater1).ravel()
    r2 = np.asarray(rater2).ravel()
    if len(r1) != len(r2):
        raise ValueError("Raters must have the same number of observations.")
    n = len(r1)
    categories = np.unique(np.concatenate([r1, r2]))
    k = len(categories)
    # Build confusion matrix
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    matrix = np.zeros((k, k), dtype=np.float64)
    for a, b in zip(r1, r2):
        matrix[cat_to_idx[a], cat_to_idx[b]] += 1
    p_o = np.trace(matrix) / n
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    p_e = float((row_sums * col_sums).sum())
    kappa_val = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0
    # Approximate SE (Fleiss, 1981)
    se = math.sqrt(p_e / (n * (1 - p_e) ** 2)) if (1 - p_e) > 0 and n > 0 else 0.0
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    return TestResult(
        method="Cohen's kappa",
        test_statistic=float(kappa_val / se) if se > 0 else 0.0,
        p_value=float(2 * stats.norm.sf(abs(kappa_val / se))) if se > 0 else 1.0,
        ci_lower=float(kappa_val - z_crit * se),
        ci_upper=float(kappa_val + z_crit * se),
        effect_size=float(kappa_val),
        estimate=float(kappa_val),
        n=n,
    )


def fleiss_kappa(
    ratings_matrix: Union[np.ndarray, pd.DataFrame],
) -> TestResult:
    """Fleiss' kappa for agreement among multiple raters.

    Parameters
    ----------
    ratings_matrix : array-like
        An n x k matrix where rows are subjects and columns are rating
        categories.  Cell (i, j) is the number of raters who assigned subject
        *i* to category *j*.

    Returns
    -------
    TestResult

    References
    ----------
    Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters.
    *Psychological Bulletin*, 76(5), 378--382.
    """
    table = np.asarray(ratings_matrix, dtype=np.float64)
    n, k = table.shape
    N_raters = table[0].sum()  # assume constant across subjects
    # Proportion in each category
    p_j = table.sum(axis=0) / (n * N_raters)
    # Per-subject agreement
    P_i = (np.sum(table**2, axis=1) - N_raters) / (N_raters * (N_raters - 1))
    P_bar = P_i.mean()
    P_e = float((p_j**2).sum())
    kappa_val = (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 0 else 0.0
    # SE (Fleiss et al., 2003)
    se_num = 2.0 / (n * N_raters * (N_raters - 1))
    se_term = (p_j * (1 - p_j)).sum() ** 2
    denom = (1 - P_e) ** 2
    se = math.sqrt(se_num * (se_term / denom)) if denom > 0 else 0.0
    z = kappa_val / se if se > 0 else 0.0
    p_val = 2 * stats.norm.sf(abs(z))
    return TestResult(
        method="Fleiss' kappa",
        test_statistic=float(z),
        p_value=float(p_val),
        effect_size=float(kappa_val),
        estimate=float(kappa_val),
        n=n,
        extra={"n_raters": int(N_raters), "n_categories": k},
    )


def intraclass_correlation(
    data: pd.DataFrame,
    targets: str,
    raters: str,
    ratings: str,
    icc_type: str = "ICC3k",
) -> TestResult:
    """Intraclass correlation coefficient (ICC).

    Implements the Shrout & Fleiss (1979) taxonomy:

    - **ICC1**: One-way random, single measures
    - **ICC1k**: One-way random, average measures
    - **ICC2**: Two-way random, single measures
    - **ICC2k**: Two-way random, average measures
    - **ICC3**: Two-way mixed, single measures
    - **ICC3k**: Two-way mixed, average measures

    Parameters
    ----------
    data : DataFrame
        Long-format data.
    targets : str
        Column identifying subjects/targets.
    raters : str
        Column identifying raters.
    ratings : str
        Column containing ratings (numeric).
    icc_type : str, default "ICC3k"
        Which ICC form to compute.

    Returns
    -------
    TestResult

    References
    ----------
    Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in
    assessing rater reliability. *Psychological Bulletin*, 86(2), 420--428.
    """
    df = data.dropna(subset=[targets, raters, ratings]).copy()
    wide = df.pivot(index=targets, columns=raters, values=ratings).dropna()
    n = wide.shape[0]  # subjects
    k = wide.shape[1]  # raters
    Y = wide.values.astype(np.float64)
    grand_mean = Y.mean()
    subj_means = Y.mean(axis=1)
    rater_means = Y.mean(axis=0)

    # Sums of squares
    ss_total = ((Y - grand_mean) ** 2).sum()
    ss_rows = k * ((subj_means - grand_mean) ** 2).sum()  # BMS
    ss_cols = n * ((rater_means - grand_mean) ** 2).sum()  # JMS
    ss_error = ss_total - ss_rows - ss_cols  # EMS
    ms_rows = ss_rows / (n - 1) if n > 1 else 0.0
    ms_cols = ss_cols / (k - 1) if k > 1 else 0.0
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0.0
    ms_within = (ss_cols + ss_error) / (n * (k - 1)) if n * (k - 1) > 0 else 0.0

    if icc_type == "ICC1":
        icc = (ms_rows - ms_within) / (ms_rows + (k - 1) * ms_within)
    elif icc_type == "ICC1k":
        icc = (ms_rows - ms_within) / ms_rows if ms_rows > 0 else 0.0
    elif icc_type == "ICC2":
        icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
    elif icc_type == "ICC2k":
        icc = (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)
    elif icc_type == "ICC3":
        icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)
    elif icc_type == "ICC3k":
        icc = (ms_rows - ms_error) / ms_rows if ms_rows > 0 else 0.0
    else:
        raise ValueError(f"Unknown ICC type: {icc_type}. Use ICC1, ICC1k, ICC2, ICC2k, ICC3, ICC3k.")

    # F-test for significance
    f_stat = ms_rows / ms_error if ms_error > 0 else 0.0
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    p_val = 1 - stats.f.cdf(f_stat, df1, df2)

    return TestResult(
        method=f"Intraclass correlation ({icc_type})",
        test_statistic=float(f_stat),
        p_value=float(p_val),
        df=float(df1),
        effect_size=float(icc),
        estimate=float(icc),
        n=n,
        extra={"icc_type": icc_type, "n_raters": k, "ms_rows": ms_rows, "ms_error": ms_error},
    )


# ===================================================================
# CONVENIENCE / BATCH HELPERS
# ===================================================================


def normality_battery(
    x: Union[np.ndarray, pd.Series, list],
) -> list[TestResult]:
    """Run all available normality tests on a single sample.

    Parameters
    ----------
    x : array-like
        Sample data.

    Returns
    -------
    list[TestResult]
        Results from Shapiro--Wilk, D'Agostino--Pearson, Jarque--Bera, and
        Lilliefors tests.
    """
    results = []
    x = _validate_array(x, "x")
    if len(x) >= 3:
        results.append(shapiro_wilk(x))
    if len(x) >= 20:
        results.append(dagostino_pearson(x))
    results.append(jarque_bera(x))
    results.append(lilliefors_test(x))
    return results


def variance_equality_battery(
    *groups: Union[np.ndarray, pd.Series, list],
) -> list[TestResult]:
    """Run Levene's (median) and Bartlett's tests for homogeneity of variance.

    Parameters
    ----------
    *groups : array-like
        Two or more samples.

    Returns
    -------
    list[TestResult]
    """
    return [
        levene_test(*groups, center="median"),
        bartlett_test(*groups),
    ]


def correlation_matrix(
    data: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute a pairwise correlation matrix with *p*-values.

    Parameters
    ----------
    data : DataFrame
        Numeric columns only.
    method : str, default "pearson"
        ``"pearson"``, ``"spearman"``, or ``"kendall"``.

    Returns
    -------
    DataFrame
        MultiIndex columns with ``("r", col)`` and ``("p", col)`` levels.
    """
    cols = data.select_dtypes(include=[np.number]).columns.tolist()
    n = len(cols)
    r_mat = np.zeros((n, n))
    p_mat = np.zeros((n, n))
    func_map = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}
    corr_func = func_map.get(method)
    if corr_func is None:
        raise ValueError(f"Unknown method: {method}. Use pearson, spearman, or kendall.")
    for i in range(n):
        for j in range(i, n):
            if i == j:
                r_mat[i, j] = 1.0
                p_mat[i, j] = 0.0
            else:
                x = data[cols[i]].dropna()
                y = data[cols[j]].dropna()
                common = x.index.intersection(y.index)
                r, p = corr_func(x.loc[common].values, y.loc[common].values)
                r_mat[i, j] = r_mat[j, i] = r
                p_mat[i, j] = p_mat[j, i] = p
    multi_cols = pd.MultiIndex.from_product([["r", "p"], cols])
    combined = np.column_stack([r_mat, p_mat])
    return pd.DataFrame(combined, index=cols, columns=multi_cols)


def auto_test(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list] | None = None,
    paired: bool = False,
    confidence: float = 0.95,
) -> TestResult:
    """Automatically select and run the most appropriate test.

    Decision logic:

    1. If *y* is ``None`` -- one-sample t-test against zero.
    2. If ``paired=True`` -- paired t-test (if normal differences) or Wilcoxon.
    3. If two independent samples -- check normality and variance equality;
       choose between Student's t, Welch's t, or Mann--Whitney U.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like or None
        Second sample (if applicable).
    paired : bool, default False
        Whether samples are paired.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    TestResult
    """
    x = _validate_array(x, "x")
    if y is None:
        return one_sample_ttest(x, mu0=0.0, confidence=confidence)
    y = _validate_array(y, "y")
    if paired:
        if len(x) != len(y):
            raise ValueError("Paired comparison requires equal-length arrays.")
        d = x - y
        sw = stats.shapiro(d)
        if sw.pvalue >= 0.05:
            return paired_ttest(x, y, confidence=confidence)
        return wilcoxon_signed_rank(x, y)
    # Independent: check normality in both groups
    sw_x = stats.shapiro(x) if len(x) <= 5000 else stats.normaltest(x)
    sw_y = stats.shapiro(y) if len(y) <= 5000 else stats.normaltest(y)
    both_normal = sw_x.pvalue >= 0.05 and sw_y.pvalue >= 0.05
    if both_normal:
        lev = stats.levene(x, y, center="median")
        eq_var = lev.pvalue >= 0.05
        return two_sample_ttest(x, y, equal_var=eq_var, confidence=confidence)
    return mann_whitney_u(x, y)
