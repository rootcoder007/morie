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

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn import _stats_core as stats

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
    d = data.dropna(subset=[outcome, factor_a, factor_b])
    yv = [float(v) for v in d[outcome].tolist()]
    fa = [str(v) for v in d[factor_a].tolist()]
    fb = [str(v) for v in d[factor_b].tolist()]
    n = len(yv)
    la, lb = sorted(set(fa)), sorted(set(fb))
    # treatment-coded design columns; type II sums of squares are
    # differences of residual sums of squares of nested fits (car::Anova)
    da = [[1.0 if v == lev else 0.0 for v in fa] for lev in la[1:]]
    db = [[1.0 if v == lev else 0.0 for v in fb] for lev in lb[1:]]
    dab = [[x * z for x, z in zip(ca, cb)] for ca in da for cb in db]

    def rss(cols):
        X = np.array([[1.0] + [c[i] for c in cols] for i in range(n)])
        beta = np.linalg.lstsq(X, np.array(yv), rcond=None)[0]
        fit = X @ beta
        return float(sum((yv[i] - float(fit[i])) ** 2 for i in range(n))), int(np.linalg.matrix_rank(X))

    rss_a, _ = rss(da)
    rss_b, _ = rss(db)
    rss_ab, _ = rss(da + db)
    rss_full, rank_full = rss(da + db + dab)
    df_res = n - rank_full
    ms_res = rss_full / df_res if df_res > 0 else float("nan")
    rows = {
        f"C({factor_a})": (rss_b - rss_ab, len(la) - 1),
        f"C({factor_b})": (rss_a - rss_ab, len(lb) - 1),
        f"C({factor_a}):C({factor_b})": (rss_ab - rss_full, rank_full - 1 - (len(la) - 1) - (len(lb) - 1)),
    }
    table = {"sum_sq": {}, "df": {}, "F": {}, "PR(>F)": {}}
    for key, (ss, dfk) in rows.items():
        f_val = (ss / dfk) / ms_res if dfk > 0 and ms_res > 0 else float("nan")
        table["sum_sq"][key] = float(ss)
        table["df"][key] = float(dfk)
        table["F"][key] = float(f_val)
        table["PR(>F)"][key] = float(stats.f.sf(f_val, dfk, df_res)) if dfk > 0 and ms_res > 0 else float("nan")
    table["sum_sq"]["Residual"] = float(rss_full)
    table["df"]["Residual"] = float(df_res)
    table["F"]["Residual"] = float("nan")
    table["PR(>F)"]["Residual"] = float("nan")
    interaction_key = f"C({factor_a}):C({factor_b})"
    ss_total = sum(table["sum_sq"].values())
    return TestResult(
        method="Two-way ANOVA",
        test_statistic=table["F"][interaction_key],
        p_value=table["PR(>F)"][interaction_key],
        effect_size=float(table["sum_sq"][f"C({factor_a})"] / ss_total),
        n=n,
        extra={"anova_table": table},
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
    p_unc = float(stats.f.sf(f_stat, df_cond, df_error))
    # Greenhouse-Geisser epsilon from the double-centred covariance of the
    # conditions: (tr S)^2 / ((k - 1) sum S_ij^2), bounded below by 1/(k-1)
    Y = [[float(v) for v in row] for row in wide.values.tolist()]
    cm = [sum(Y[i][j] for i in range(n)) / n for j in range(k)]
    S = [[sum((Y[i][a] - cm[a]) * (Y[i][b] - cm[b]) for i in range(n)) / (n - 1) for b in range(k)] for a in range(k)]
    rmean = [sum(r) / k for r in S]
    gm = sum(rmean) / k
    D = [[S[a][b] - rmean[a] - rmean[b] + gm for b in range(k)] for a in range(k)]
    ssq = sum(v * v for r in D for v in r)
    eps = (sum(D[a][a] for a in range(k)) ** 2 / ((k - 1) * ssq)) if ssq > 0 else 1.0
    eps = min(1.0, max(eps, 1.0 / (k - 1)))
    p = float(stats.f.sf(f_stat, eps * df_cond, eps * df_error))
    eta2 = ss_cond / (ss_cond + ss_error) if (ss_cond + ss_error) > 0 else 0.0
    return TestResult(
        method="Repeated-measures ANOVA",
        test_statistic=float(f_stat),
        p_value=p,
        df=float(df_cond),
        effect_size=float(eta2),
        n=n,
        extra={
            "df_error": df_error,
            "ss_cond": ss_cond,
            "ss_error": ss_error,
            "epsilon_gg": float(eps),
            "p_uncorrected": p_unc,
        },
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
    exp = np.full_like(obs, obs.sum() / len(obs)) if expected is None else np.asarray(expected, dtype=np.float64)
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
        chi2_stat = float(b)  # binom.test(b, b + c): successes among discordant pairs
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
    p = float(stats.chi2.sf(q_stat, k - 1))
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
    if n < 1290 and len(set(x.tolist())) == n and len(set(y.tolist())) == n:
        # cor.test(method = "spearman"): exact (n <= 9) or Edgeworth
        # (AS 89) p-value for untied data; the t approximation otherwise
        from morie.inference import _prho

        q = (n**3 - n) * (1.0 - rho) / 6.0
        pp = _prho(n, round(q) + 0.0, False) if q > (n**3 - n) / 6.0 else _prho(n, round(q) + 2.0, True)
        p = min(2.0 * pp, 1.0)
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
    r, _ = stats.pearsonr(res_x, res_y)
    p_vars = Z.shape[1]
    # t test on n - 2 - p df and Fisher z with n - 3 - p, as ppcor::pcor.test
    df_val = n - 2 - p_vars
    t_stat = r * math.sqrt(df_val / (1 - r * r)) if df_val > 0 and abs(r) < 1 else math.copysign(math.inf, r)
    p = 2.0 * float(stats.t.sf(abs(t_stat), df_val)) if df_val > 0 else float("nan")
    z = np.arctanh(r)
    se_z = 1.0 / math.sqrt(df_val - 1) if df_val > 1 else np.inf
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
    r, _ = stats.pearsonr(res_x, y)
    # t test on n - 2 - p df, as ppcor::spcor.test
    df_val = n - 2 - Z.shape[1]
    t_stat = r * math.sqrt(df_val / (1 - r * r)) if df_val > 0 and abs(r) < 1 else math.copysign(math.inf, r)
    p = 2.0 * float(stats.t.sf(abs(t_stat), df_val)) if df_val > 0 else float("nan")
    return TestResult(
        method="Semi-partial correlation",
        test_statistic=float(r),
        p_value=float(p),
        df=float(df_val),
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
        Effect size is rank-biserial correlation :math:`r = 2U / (n_1 n_2) - 1`,
        positive when *x* tends to exceed *y*. The p-value is exact for
        untied samples both under 50, as ``wilcox.test``, and the
        tie-corrected normal approximation otherwise.
    """
    x = _validate_array(x, "x")
    y = _validate_array(y, "y")
    nx, ny = len(x), len(y)
    xy = x.tolist() + y.tolist()
    exact = nx < 50 and ny < 50 and len(set(xy)) == len(xy)
    u_stat, p = stats.mannwhitneyu(x, y, alternative=alternative, method="exact" if exact else "asymptotic")
    r_rb = 2.0 * u_stat / (nx * ny) - 1.0 if nx * ny > 0 else 0.0
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

_GOF_LILLIE_N = (4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 75, 100)
_GOF_LILLIE_ALPHA = (0.100, 0.050, 0.010, 0.001)

_GOF_LILLIE_NORM = (
    (0.344, 0.375, 0.414, 0.432),
    (0.320, 0.344, 0.398, 0.427),
    (0.298, 0.323, 0.369, 0.421),
    (0.281, 0.305, 0.351, 0.399),
    (0.266, 0.289, 0.334, 0.383),
    (0.252, 0.273, 0.316, 0.366),
    (0.240, 0.261, 0.305, 0.350),
    (0.231, 0.251, 0.291, 0.331),
    (0.223, 0.242, 0.281, 0.327),
    (0.208, 0.226, 0.262, 0.302),
    (0.195, 0.213, 0.249, 0.291),
    (0.185, 0.201, 0.234, 0.272),
    (0.176, 0.192, 0.223, 0.266),
    (0.159, 0.173, 0.202, 0.236),
    (0.146, 0.159, 0.186, 0.219),
    (0.127, 0.139, 0.161, 0.190),
    (0.114, 0.125, 0.145, 0.173),
    (0.105, 0.114, 0.133, 0.159),
    (0.094, 0.102, 0.119, 0.138),
    (0.082, 0.089, 0.104, 0.121),
)
_GOF_LILLIE_EXP = (
    (0.444, 0.483, 0.556, 0.626),
    (0.405, 0.443, 0.514, 0.585),
    (0.374, 0.410, 0.477, 0.551),
    (0.347, 0.381, 0.444, 0.509),
    (0.327, 0.359, 0.421, 0.502),
    (0.310, 0.339, 0.399, 0.460),
    (0.296, 0.325, 0.379, 0.444),
    (0.284, 0.312, 0.366, 0.433),
    (0.271, 0.299, 0.350, 0.412),
    (0.252, 0.277, 0.325, 0.388),
    (0.237, 0.261, 0.311, 0.366),
    (0.224, 0.247, 0.293, 0.328),
    (0.213, 0.234, 0.279, 0.329),
    (0.192, 0.211, 0.251, 0.296),
    (0.176, 0.193, 0.229, 0.270),
    (0.153, 0.168, 0.201, 0.241),
    (0.137, 0.150, 0.179, 0.214),
    (0.125, 0.138, 0.164, 0.193),
    (0.113, 0.124, 0.146, 0.173),
    (0.098, 0.108, 0.127, 0.150),
)
_GOF_LILLIE_ASYMP = {
    "norm": (0.816, 0.888, 1.038, 1.212),
    "expon": (0.980, 1.077, 1.274, 1.501),
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
    return np.array([np.interp(n, _GOF_LILLIE_N, tab[:, j]) for j in range(tab.shape[1])])


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


def _lillie_p_norm(k: float, n: int) -> float:
    """Lilliefors normal-null p-value: Dallal & Wilkinson (1986) below 0.1,
    Stephens' (1974) modified-statistic polynomials above, as
    nortest::lillie.test."""
    kd, nd = (k, n) if n <= 100 else (k * (n / 100) ** 0.49, 100)
    p = math.exp(
        -7.01256 * kd**2 * (nd + 2.78019)
        + 2.99587 * kd * math.sqrt(nd + 2.78019)
        - 0.122119
        + 0.974598 / math.sqrt(nd)
        + 1.67997 / nd
    )
    if p > 0.1:
        kk = (math.sqrt(n) - 0.01 + 0.85 / math.sqrt(n)) * k
        if kk <= 0.302:
            p = 1.0
        elif kk <= 0.5:
            p = 2.76773 - 19.828315 * kk + 80.709644 * kk**2 - 138.55152 * kk**3 + 81.218052 * kk**4
        elif kk <= 0.9:
            p = -4.901232 + 40.662806 * kk - 97.490286 * kk**2 + 94.029866 * kk**3 - 32.355711 * kk**4
        elif kk <= 1.31:
            p = 6.198765 - 19.558097 * kk + 23.186922 * kk**2 - 12.234627 * kk**3 + 2.423045 * kk**4
        else:
            p = 0.0
    return float(p)


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
    if dist == "norm":
        # D'Agostino & Stephens (1986, Table 4.9), as nortest::ad.test
        aa = astar
        if aa < 0.2:
            p = 1 - math.exp(-13.436 + 101.14 * aa - 223.73 * aa**2)
        elif aa < 0.34:
            p = 1 - math.exp(-8.318 + 42.796 * aa - 59.938 * aa**2)
        elif aa < 0.6:
            p = math.exp(0.9177 - 4.279 * aa - 1.38 * aa**2)
        elif aa < 10:
            p = math.exp(1.2937 - 5.709 * aa + 0.0186 * aa**2)
        else:
            p = 3.7e-24
        p, bounded = float(p), None
    else:
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
    if dist == "norm":
        p, bounded = _lillie_p_norm(d, n), None
    else:
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
    _, p = stats.fisher_exact(table, alternative=alternative)
    n = int(table.sum())
    # conditional MLE and exact interval, as fisher.test: two-sided 95%, or
    # the one-sided 95% bound (the matching limit of the two-sided 90%)
    from morie.inference import _fisher_conditional

    tab = [[int(v) for v in r] for r in table.tolist()]
    est, lo, hi = _fisher_conditional(tab, 0.95 if alternative == "two-sided" else 0.90)
    if alternative == "greater":
        hi = math.inf
    elif alternative == "less":
        lo = 0.0
    return TestResult(
        method="Fisher's exact test",
        test_statistic=float(est),
        p_value=float(p),
        ci_lower=float(lo),
        ci_upper=float(hi),
        estimate=float(est),
        n=n,
        extra={
            "sample_odds_ratio": float(table[0, 0] * table[1, 1] / (table[0, 1] * table[1, 0]))
            if table[0, 1] * table[1, 0] > 0
            else math.inf
        },
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
    # Fleiss, Cohen and Everitt (1969): the null variance for the z test
    # (as irr::kappa2) and the non-null variance for the interval (as
    # psych::cohen.kappa)
    P = matrix / n
    rs = [float(v) for v in row_sums]
    cs = [float(v) for v in col_sums]
    den = n * (1 - p_e) ** 2
    v0 = (p_e + p_e**2 - sum(r * c * (r + c) for r, c in zip(rs, cs))) / den if den > 0 else 0.0
    diag_part = sum(float(P[i, i]) * (1 - (rs[i] + cs[i]) * (1 - kappa_val)) ** 2 for i in range(k))
    off_part = sum(float(P[i, j]) * (cs[i] + rs[j]) ** 2 for i in range(k) for j in range(k) if i != j)
    v1 = (
        (diag_part + (1 - kappa_val) ** 2 * off_part - (kappa_val - p_e * (1 - kappa_val)) ** 2) / den
        if den > 0
        else 0.0
    )
    se = math.sqrt(max(v1, 0.0))
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    z_stat = kappa_val / math.sqrt(v0) if v0 > 0 else 0.0
    return TestResult(
        method="Cohen's kappa",
        test_statistic=float(z_stat),
        p_value=float(2 * stats.norm.sf(abs(z_stat))) if v0 > 0 else 1.0,
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
    # null SE (Fleiss, Levin & Paik 2003, eq. 18.15), as irr::kappam.fleiss
    pq = [float(v) * (1 - float(v)) for v in p_j]
    spq = sum(pq)
    inner = spq**2 - sum(a * (1 - 2 * float(v)) for a, v in zip(pq, p_j))
    se = (
        math.sqrt(2) / (spq * math.sqrt(n * N_raters * (N_raters - 1))) * math.sqrt(inner)
        if spq > 0 and inner > 0
        else 0.0
    )
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

    # F tests and intervals of Shrout & Fleiss (1979) / McGraw & Wong (1996),
    # as psych::ICC (95%)
    df1 = n - 1
    fq = stats.f.ppf
    if icc_type in ("ICC1", "ICC1k"):
        df2 = n * (k - 1)
        f_stat = ms_rows / ms_within if ms_within > 0 else 0.0
        fl = f_stat / fq(0.975, df1, df2)
        fu = f_stat * fq(0.975, df2, df1)
        lo, hi = ((fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1)) if icc_type == "ICC1" else (1 - 1 / fl, 1 - 1 / fu)
    else:
        df2 = (n - 1) * (k - 1)
        f_stat = ms_rows / ms_error if ms_error > 0 else 0.0
        if icc_type in ("ICC3", "ICC3k"):
            fl = f_stat / fq(0.975, df1, df2)
            fu = f_stat * fq(0.975, df2, df1)
            lo, hi = (
                ((fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1)) if icc_type == "ICC3" else (1 - 1 / fl, 1 - 1 / fu)
            )
        else:
            i2 = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
            fj = ms_cols / ms_error
            vn = (k - 1) * (n - 1) * (k * i2 * fj + n * (1 + (k - 1) * i2) - k * i2) ** 2
            vd = (n - 1) * k**2 * i2**2 * fj**2 + (n * (1 + (k - 1) * i2) - k * i2) ** 2
            v = vn / vd
            f3u = fq(0.975, n - 1, v)
            f3l = fq(0.975, v, n - 1)
            lo = n * (ms_rows - f3u * ms_error) / (f3u * (k * ms_cols + (k * n - k - n) * ms_error) + n * ms_rows)
            hi = n * (f3l * ms_rows - ms_error) / (k * ms_cols + (k * n - k - n) * ms_error + n * f3l * ms_rows)
            if icc_type == "ICC2k":
                lo, hi = lo * k / (1 + lo * (k - 1)), hi * k / (1 + hi * (k - 1))
    p_val = float(stats.f.sf(f_stat, df1, df2))
    return TestResult(
        method=f"Intraclass correlation ({icc_type})",
        test_statistic=float(f_stat),
        p_value=float(p_val),
        df=float(df1),
        ci_lower=float(lo),
        ci_upper=float(hi),
        effect_size=float(icc),
        estimate=float(icc),
        n=n,
        extra={"icc_type": icc_type, "n_raters": k, "ms_rows": ms_rows, "ms_error": ms_error, "df2": df2},
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
