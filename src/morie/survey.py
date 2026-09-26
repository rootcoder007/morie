"""
Survey-weighted estimation wrappers.
Parallels the ``survey`` and ``srvyr`` packages in R.

This module provides:

1. :class:`SurveyDesign` -- encapsulates survey data with weights and strata.
2. Horvitz-Thompson and Hájek estimators for population totals and means.
3. Ratio estimator for auxiliary-variable-assisted estimation.
4. Post-stratification and calibration (raking) weight adjustments.
5. Domain/subpopulation estimation with design-based standard errors.
6. Complex-survey GLM with weights, clustering, and stratification.

Statistical assumptions and caveats
-------------------------------------
- All functions assume the supplied weights are probability/analytic weights
  (i.e., w_i = 1 / pi_i where pi_i is the first-order inclusion probability).
  They are NOT frequency or expansion weights.
- Standard errors under complex sampling are approximated by Taylor
  linearisation or the jackknife/bootstrap where noted.
- For production survey analysis, validate against the R ``survey`` package
  (Lumley, 2010) or Stata's ``svy`` prefix.

References
----------
Lumley, T. (2010). Complex Surveys: A Guide to Analysis Using R. Wiley.
Cochran, W. G. (1977). Sampling Techniques (3rd ed.). Wiley.
Horvitz, D. G., & Thompson, D. J. (1952). A generalisation of sampling without
    replacement from a finite universe. JASA, 47(260), 663-685.
Hájek, J. (1971). Comment on "An essay on the logical foundations of survey
    sampling." In Godambe & Sprott (Eds.), Foundations of Statistical Inference.
    Holt, Rinehart and Winston.
"""

import math
import warnings

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn import _glm_core
from morie.fn import _stats_core as scipy_stats

# ---------------------------------------------------------------------------
# Native formula interface for weighted GLMs.
#
# survey.py used to call statsmodels' formula API (smf.glm). That dependency
# is gone, so the formula is parsed here and handed to morie.fn._glm_core.glm,
# which fits by IRLS and reports the sandwich standard errors. Weights are
# ANALYTIC (variance-scaling) weights, the correct treatment for unequal
# probability sampling: df_resid stays n - k rather than sum(w) - k.
# ---------------------------------------------------------------------------


def _native_glm_from_formula(formula, data, family, weights=None):
    """Fit a weighted GLM from a formula.

    The formula parsing, the design build and the result wrapper all
    live in morie.fn._glm_formula, which is the one native replacement
    for statsmodels' formula API. survey.py had its own copy of all
    three; keeping a second copy is how the two drift apart.
    """
    from morie.fn import _glm_formula

    w = None if weights is None else [float(v) for v in weights]
    return _glm_formula.glm(formula, data, family=family, weights=w).fit()


def _svy_glm(formula, data, family, weights, psu=None, strata=None):
    """Design-based GLM as survey::svyglm: the weighted IRLS fit and the
    linearisation (sandwich) variance A^-1 M A^-1, where A is the weighted
    information and M the between-PSU variance of the score totals within
    strata, scaled by n_h/(n_h - 1); t tests on degf + 1 - p df, degf the
    number of PSUs less the number of strata (Binder 1983; Lumley 2010,
    sec. 5.2)."""
    from morie.fn import _glm_formula

    fam = family
    if not isinstance(fam, str):
        fam = getattr(fam, "name", None) or type(fam).__name__.lower()
    fam = str(fam).lower().replace("-", "").replace("_", "")
    w = [float(v) for v in (weights.values if hasattr(weights, "values") else weights)]
    model = _glm_formula.glm(formula, data, family=fam, weights=w)
    res = model.fit()
    F = _glm_core.FAMILIES[fam]
    beta = [float(b) for b in res.params]
    X = [[1.0] + [float(v) for v in r] if model.intercept else [float(v) for v in r] for r in model.X]
    y = [float(v) for v in model.y]
    n, k = len(y), len(beta)
    A = [[0.0] * k for _ in range(k)]
    U = []
    for i in range(n):
        eta = sum(X[i][j] * beta[j] for j in range(k))
        mu = F["linkinv"](eta)
        g = F["mu_eta"](eta)
        v = F["variance"](mu)
        U.append([w[i] * X[i][j] * (y[i] - mu) * g / v for j in range(k)])
        c = w[i] * g * g / v
        for a in range(k):
            for b in range(k):
                A[a][b] += c * X[i][a] * X[i][b]
    psu = list(range(n)) if psu is None else [v for v in (psu.values if hasattr(psu, "values") else psu)]
    strata = [0] * n if strata is None else [v for v in (strata.values if hasattr(strata, "values") else strata)]
    M = [[0.0] * k for _ in range(k)]
    n_psu = 0
    for h in sorted(set(strata), key=str):
        tot = {}
        for i in range(n):
            if strata[i] == h:
                t = tot.setdefault(psu[i], [0.0] * k)
                for j in range(k):
                    t[j] += U[i][j]
        nh = len(tot)
        n_psu += nh
        if nh < 2:
            continue
        ub = [sum(t[j] for t in tot.values()) / nh for j in range(k)]
        for t in tot.values():
            for a in range(k):
                for b in range(k):
                    M[a][b] += nh / (nh - 1) * (t[a] - ub[a]) * (t[b] - ub[b])
    Ai = _glm_core._inv(A)
    V = [[sum(Ai[a][c] * M[c][d] * Ai[d][b] for c in range(k) for d in range(k)) for b in range(k)] for a in range(k)]
    se = [math.sqrt(V[j][j]) for j in range(k)]
    df_res = n_psu - len(set(strata)) + 1 - k
    tv = [b / e if e > 0 else float("nan") for b, e in zip(beta, se)]
    pv = [2.0 * float(scipy_stats.t.sf(abs(t), df_res)) if df_res > 0 and t == t else float("nan") for t in tv]
    from morie.fn._glm_formula import _Result

    return _Result(
        beta,
        se,
        res.param_names,
        n,
        df_res,
        tvalues=tv,
        pvalues=pv,
        fittedvalues=res.fittedvalues,
        cov=V,
        extra=dict(model._fit, cov_type="survey"),
        model=model,
    )


class SurveyDesign:
    """
    Encapsulates survey data with its corresponding sampling weights and strata.
    """

    def __init__(self, data: pd.DataFrame, weights_col: str, strata_col: str | None = None):
        """
        Initialize the survey design object.

        :param data: The pandas DataFrame containing the survey data.
        :type data: pandas.DataFrame
        :param weights_col: The column name corresponding to the survey weights.
        :type weights_col: str
        :param strata_col: The column name indicating survey strata, defaults to None.
        :type strata_col: str, optional
        """
        self.data = data
        self.weights = data[weights_col]
        self.strata = data[strata_col] if strata_col else None

    def weighted_mean(self, variable: str) -> float:
        """
        Compute the survey-weighted mean of a continuous variable.

        :param variable: The name of the variable to average.
        :type variable: str
        :return: The weighted average.
        :rtype: float
        """
        y = self.data[variable]
        return float(np.average(y, weights=self.weights))

    def svyglm(self, formula: str, family=None):
        """
        Fit a survey-weighted generalized linear model.

        Survey probability weights (such as CPADS ``weight`` / ``wtpumf``) are
        probability weights, **not** frequency expansion weights. The fit is
        the weighted IRLS estimate and the standard errors are the
        design-based linearisation (sandwich) variance, with the design's
        strata and each unit its own PSU, as ``survey::svyglm``.

        :param formula: A formula string.
        :type formula: str
        :param family: Family name or object, defaults to binomial.
        :return: A fitted GLM result with design-based standard errors.

        References
        ----------
        Lumley, T. (2010). *Complex Surveys: A Guide to Analysis Using R*.
        Wiley. (Chapter 2 -- probability weights vs. frequency weights.)
        """
        if family is None:
            family = "binomial"

        # var_weights: analytic/probability weights -- correct for survey data.
        # Each weight w_i re-scales the variance of observation i by 1/w_i,
        # which is the appropriate treatment for unequal-probability sampling.
        # Do NOT use freq_weights, which expands the dataset by the weight
        # value and thus inflates n_effective and deflates standard errors.
        return _svy_glm(formula, self.data, family, self.weights, strata=self.strata)


# ===========================================================================
# SECTION 2 -- HORVITZ-THOMPSON AND HÁJEK ESTIMATORS
# ===========================================================================


def horvitz_thompson_total(
    y: np.ndarray,
    inclusion_probs: np.ndarray,
) -> dict:
    """
    Horvitz-Thompson estimator of population total.

    The HT estimator is:

    .. math::

        \\hat{\\tau}_{HT} = \\sum_{i \\in s} \\frac{y_i}{\\pi_i}

    where :math:`\\pi_i` is the first-order inclusion probability for unit i.

    The variance estimator is the Horvitz-Thompson one under Poisson sampling,
    where the pairwise inclusion probabilities are exactly
    :math:`\\pi_{ij} = \\pi_i \\pi_j` (as ``survey::svytotal`` with
    ``pps = poisson_sampling(pi)``):

    .. math::

        \\hat{V}(\\hat{\\tau}_{HT}) \\approx \\sum_i \\frac{y_i^2}{\\pi_i^2}
        \\cdot \\frac{1 - \\pi_i}{1}

    Under other without-replacement designs it ignores the pairwise
    inclusion probabilities the design actually has.

    :param y: Observed response values for sampled units (1-D array-like).
    :param inclusion_probs: First-order inclusion probabilities pi_i for each
        sampled unit (values in (0, 1]).
    :return: dict with keys ``total``, ``se``, ``ci_lower``, ``ci_upper``.
    :raises ValueError: If y and inclusion_probs have different lengths, or
        any inclusion probability is <= 0 or > 1.

    References
    ----------
    Horvitz, D. G., & Thompson, D. J. (1952). A generalisation of sampling
        without replacement from a finite universe. JASA, 47(260), 663-685.
    Cochran, W. G. (1977). Sampling Techniques (3rd ed.). Wiley. (Chapter 9A.)
    """
    y_arr = np.asarray(y, dtype=float)
    pi_arr = np.asarray(inclusion_probs, dtype=float)
    if len(y_arr) != len(pi_arr):
        raise ValueError(f"y and inclusion_probs must have the same length; got {len(y_arr)} and {len(pi_arr)}.")
    if np.any(pi_arr <= 0) or np.any(pi_arr > 1):
        raise ValueError("All inclusion_probs must be in (0, 1].")
    if len(y_arr) == 0:
        raise ValueError("y must contain at least one observation.")

    # HT total: sum of y_i / pi_i
    ht_total = float(np.sum(y_arr / pi_arr))

    # Variance approximation: SYG with pi_ij ≈ pi_i * pi_j (with-replacement approx)
    # V_HT ≈ sum_i (y_i / pi_i)^2 * (1 - pi_i)
    z = y_arr / pi_arr  # expansion values
    var_ht = float(np.sum(z**2 * (1.0 - pi_arr)))
    se = math.sqrt(max(0.0, var_ht))

    # Approximate 95% CI using normal quantile (valid for large samples)
    z_crit = float(scipy_stats.norm.ppf(0.975))
    return {
        "total": ht_total,
        "se": se,
        "ci_lower": ht_total - z_crit * se,
        "ci_upper": ht_total + z_crit * se,
    }


def hajek_mean(
    y: np.ndarray,
    weights: np.ndarray,
) -> dict:
    """
    Hájek estimator for population mean (ratio estimator).

    The Hájek estimator normalises the HT estimator by the estimated
    population size, eliminating sensitivity to total weight magnitude:

    .. math::

        \\bar{y}_H = \\frac{\\sum_i w_i y_i}{\\sum_i w_i}

    where :math:`w_i = 1 / \\pi_i` are the survey weights.

    Standard error is computed via Taylor (delta-method) linearisation:

    .. math::

        \\hat{V}(\\bar{y}_H) \\approx \\frac{1}{N^2} \\sum_i w_i^2 (y_i - \\bar{y}_H)^2
        \\cdot \\frac{1 - \\pi_i}{\\pi_i}

    For the with-replacement approximation (pi_i = w_i / N_hat) this simplifies
    to the weighted sample variance divided by the effective sample size.

    :param y: Response values (1-D array-like).
    :param weights: Survey weights w_i = 1/pi_i (must be > 0).
    :return: dict with keys ``mean``, ``se``, ``ci_lower``, ``ci_upper``.
    :raises ValueError: If y and weights have different lengths or weights <= 0.

    References
    ----------
    Hájek, J. (1971). Comment in Godambe & Sprott (Eds.), Foundations of
        Statistical Inference. Holt, Rinehart and Winston.
    Cochran, W. G. (1977). Sampling Techniques (3rd ed.). Wiley. (Section 6.13.)
    """
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    if len(y_arr) != len(w_arr):
        raise ValueError(f"y and weights must have the same length; got {len(y_arr)} and {len(w_arr)}.")
    if np.any(w_arr <= 0):
        raise ValueError("All weights must be > 0.")
    if len(y_arr) < 2:
        raise ValueError("At least 2 observations are required for SE computation.")

    sum_w = float(np.sum(w_arr))
    hajek_mean_val = float(np.sum(w_arr * y_arr) / sum_w)

    # Taylor linearisation SE: weighted variance of residuals scaled by N_hat^2
    residuals = y_arr - hajek_mean_val
    # With-replacement approximation: treat pi_i ≈ w_i / sum_w => (1 - pi_i) ≈ 1
    # V(y_bar_H) ≈ (1/sum_w^2) * sum(w_i^2 * (y_i - y_bar_H)^2)
    # with the n/(n - 1) of the with-replacement variance, as survey::svymean
    n = len(y_arr)
    var_hajek = n / (n - 1) * float(np.sum(w_arr**2 * residuals**2)) / (sum_w**2)
    se = math.sqrt(max(0.0, var_hajek))

    z_crit = float(scipy_stats.norm.ppf(0.975))
    return {
        "mean": hajek_mean_val,
        "se": se,
        "ci_lower": hajek_mean_val - z_crit * se,
        "ci_upper": hajek_mean_val + z_crit * se,
    }


# ===========================================================================
# SECTION 3 -- RATIO ESTIMATOR
# ===========================================================================


def ratio_estimator(
    y: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
    X_population_total: float,
) -> dict:
    """
    Survey ratio estimator for a population total.

    The ratio estimator exploits auxiliary information X whose population
    total is known:

    .. math::

        \\hat{Y}_R = \\hat{R} \\cdot X_{\\text{pop}}

    where :math:`\\hat{R} = \\hat{Y}_{HT} / \\hat{X}_{HT}` is the estimated
    ratio of totals.

    Standard error via Taylor linearisation:

    .. math::

        \\hat{V}(\\hat{Y}_R) \\approx \\frac{1}{\\hat{X}_{HT}^2}
        \\sum_i w_i^2 (y_i - \\hat{R} x_i)^2

    :param y: Response variable values for sampled units (1-D array-like).
    :param x: Auxiliary variable values for sampled units (same length as y).
    :param weights: Survey weights w_i (must be > 0).
    :param X_population_total: Known population total of the auxiliary variable.
    :return: dict with keys ``ratio``, ``total_estimate``, ``se``,
        ``ci_lower``, ``ci_upper``.
    :raises ValueError: If inputs have different lengths, weights <= 0, or
        X_population_total <= 0.

    References
    ----------
    Cochran, W. G. (1977). Sampling Techniques (3rd ed.). Wiley. (Section 6.4.)
    Lumley, T. (2010). Complex Surveys: A Guide to Analysis Using R. Wiley.
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    if len(y_arr) != len(x_arr) or len(y_arr) != len(w_arr):
        raise ValueError("y, x, and weights must all have the same length.")
    if np.any(w_arr <= 0):
        raise ValueError("All weights must be > 0.")
    if X_population_total <= 0:
        raise ValueError(f"X_population_total must be > 0, got {X_population_total}.")

    y_ht = float(np.sum(w_arr * y_arr))
    x_ht = float(np.sum(w_arr * x_arr))

    if x_ht == 0:
        raise ValueError("Weighted sum of auxiliary variable x is zero.")

    r_hat = y_ht / x_ht
    total_estimate = r_hat * X_population_total

    # Taylor linearisation SE
    residuals = y_arr - r_hat * x_arr
    # with the n/(n - 1) of the with-replacement variance, as survey::svyratio
    n = len(y_arr)
    var_est = n / (n - 1) * float(np.sum(w_arr**2 * residuals**2)) / (x_ht**2) * (X_population_total**2)
    se = math.sqrt(max(0.0, var_est))

    z_crit = float(scipy_stats.norm.ppf(0.975))
    return {
        "ratio": float(r_hat),
        "total_estimate": float(total_estimate),
        "se": float(se),
        "ci_lower": float(total_estimate - z_crit * se),
        "ci_upper": float(total_estimate + z_crit * se),
    }


# ===========================================================================
# SECTION 4 -- WEIGHT CALIBRATION
# ===========================================================================


def poststratification_weights(
    df: pd.DataFrame,
    strata_col: str,
    population_counts: dict[str, int],
) -> pd.Series:
    """
    Compute post-stratification weights so the sample distribution matches
    the known population distribution.

    For each stratum h:

    .. math::

        w_i^{\\text{PS}} = \\frac{N_h / N}{n_h / n}

    where :math:`N_h` is the known population count in stratum h, :math:`N`
    is the total population, :math:`n_h` is the sample count in stratum h,
    and :math:`n` is the total sample size.

    :param df: Input DataFrame.
    :param strata_col: Column name identifying strata.
    :param population_counts: Dict mapping each stratum label (as string) to
        its population count.
    :return: pd.Series of post-stratification weights, indexed like df.
    :raises ValueError: If strata_col not in df, any stratum in df is absent
        from population_counts, or sample stratum count is zero.

    References
    ----------
    Lumley, T. (2010). Complex Surveys: A Guide to Analysis Using R. Wiley. (Chapter 7.)
    Little, R. J. A. (1993). Post-stratification: A modeler's perspective.
        JASA, 88(423), 1001-1012.
    """
    if strata_col not in df.columns:
        raise ValueError(f"Column '{strata_col}' not found in DataFrame.")
    strata_vals = df[strata_col].astype(str)
    missing_strata = set(strata_vals.unique()) - set(str(k) for k in population_counts)
    if missing_strata:
        raise ValueError(f"Strata {missing_strata} appear in the sample but are missing from population_counts.")

    N_total = sum(population_counts.values())
    n_total = len(df)

    weights = pd.Series(np.ones(len(df), dtype=float), index=df.index)
    for stratum, N_h in population_counts.items():
        stratum_str = str(stratum)
        mask = strata_vals == stratum_str
        n_h = int(mask.sum())
        if n_h == 0:
            warnings.warn(
                f"Stratum '{stratum}' has 0 observations in the sample; "
                "post-stratification weight is undefined for this stratum.",
                stacklevel=2,
            )
            continue
        pop_frac = N_h / N_total
        samp_frac = n_h / n_total
        weights[mask] = pop_frac / samp_frac
    return weights


def calibration_weights(
    df: pd.DataFrame,
    aux_vars: list[str],
    population_totals: dict[str, float],
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> pd.Series:
    """
    Raking calibration to known population totals of the auxiliary
    variables (Deville & Sarndal 1992, raking distance).

    For each auxiliary variable x_j, the calibration adjusts weights
    multiplicatively so that:

    .. math::

        \\sum_i w_i x_{ij} = T_j

    where :math:`T_j` is the known population total for variable j.

    IPF convergence is assessed by the maximum relative deviation across
    all margins at each iteration.

    :param df: Input DataFrame with auxiliary variable columns.
    :param aux_vars: List of column names to calibrate on.
    :param population_totals: Dict mapping column name to known population total.
    :param max_iter: Maximum number of IPF iterations. Default 50.
    :param tol: Convergence tolerance (maximum relative deviation). Default 1e-6.
    :return: pd.Series of calibrated weights.
    :raises ValueError: If any aux_var is missing from df or population_totals,
        or initial weights are zero.

    References
    ----------
    Deville, J.-C., & Sarndal, C.-E. (1992). Calibration estimators in survey
        sampling. JASA, 87(418), 376-382.
    Deming, W. E., & Stephan, F. F. (1940). On a least squares adjustment of
        a sampled frequency table when the expected marginal totals are known.
        Annals of Mathematical Statistics, 11(4), 427-444.
    """
    for var in aux_vars:
        if var not in df.columns:
            raise ValueError(f"Auxiliary variable '{var}' not found in DataFrame.")
        if var not in population_totals:
            raise ValueError(f"Population total for '{var}' not found in population_totals.")

    # Raking (Deville & Sarndal 1992): w_i = exp(x_i' lambda) from unit
    # starting weights, lambda solving sum_i w_i x_i = T by Newton, as
    # survey::calibrate(calfun = "raking") with no intercept
    X = [[float(v) for v in df[var].astype(float).values.tolist()] for var in aux_vars]
    T = [float(population_totals[var]) for var in aux_vars]
    n = len(df)
    k = len(aux_vars)
    lam = [0.0] * k
    w = [1.0] * n
    for _iteration in range(max_iter):
        w = [math.exp(sum(lam[j] * X[j][i] for j in range(k))) for i in range(n)]
        F = [sum(w[i] * X[j][i] for i in range(n)) - T[j] for j in range(k)]
        if max(abs(F[j]) / max(abs(T[j]), 1.0) for j in range(k)) < tol:
            break
        J = np.array([[sum(w[i] * X[a][i] * X[b][i] for i in range(n)) for b in range(k)] for a in range(k)])
        step = np.linalg.solve(J, np.array(F))
        lam = [lam[j] - float(step[j]) for j in range(k)]
    else:
        warnings.warn(f"Raking calibration did not converge within {max_iter} iterations.", stacklevel=2)
    return pd.Series(w, index=df.index)


# ===========================================================================
# SECTION 5 -- DOMAIN / SUBPOPULATION ESTIMATION
# ===========================================================================


def subpopulation_estimate(
    df: pd.DataFrame,
    domain_col: str,
    domain_value,
    outcome_col: str,
    weight_col: str,
) -> dict:
    """
    Design-based estimate for a domain (subpopulation) mean.

    Treats domain membership as a 0/1 indicator and applies the Hájek
    estimator within the full-sample design frame (i.e., does NOT subset
    then reweight, which would ignore the variance contribution from
    domain membership uncertainty).

    Domain mean:

    .. math::

        \\bar{y}_d = \\frac{\\sum_{i: d_i = 1} w_i y_i}{\\sum_{i: d_i = 1} w_i}

    SE via linearisation (Woodruff, 1971):

    .. math::

        \\hat{V}(\\bar{y}_d) \\approx \\frac{1}{N_d^2}
        \\sum_i w_i^2 z_i^2

    where :math:`z_i = d_i (y_i - \\bar{y}_d)` and
    :math:`N_d = \\sum_{i: d_i=1} w_i`.

    :param df: Full sample DataFrame (do NOT pre-filter to domain).
    :param domain_col: Column name identifying the domain.
    :param domain_value: Value of domain_col that identifies the target subpopulation.
    :param outcome_col: Column name of the outcome variable.
    :param weight_col: Column name of survey weights.
    :return: dict with keys ``mean``, ``se``, ``ci_lower``, ``ci_upper``, ``n_domain``.
    :raises ValueError: If required columns are missing.

    References
    ----------
    Woodruff, R. S. (1971). A simple method for approximating the variance of a
        complicated estimate. JASA, 66(334), 411-414.
    Lumley, T. (2010). Complex Surveys. Wiley. (Section 4.2.)
    """
    for col in [domain_col, outcome_col, weight_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    domain_mask = (df[domain_col] == domain_value).values.astype(float)
    y = df[outcome_col].astype(float).values
    w = df[weight_col].astype(float).values

    if np.any(w <= 0):
        raise ValueError("All weights must be > 0.")

    n_domain = int(domain_mask.sum())
    if n_domain == 0:
        raise ValueError(f"No observations found with {domain_col} == {domain_value!r}.")

    N_d = float(np.sum(w * domain_mask))
    y_bar_d = float(np.sum(w * domain_mask * y) / N_d)

    # Woodruff linearisation: z_i = d_i * (y_i - y_bar_d)
    z = domain_mask * (y - y_bar_d)
    # n/(n - 1) over the FULL sample, as survey::svymean on subset(design, .)
    n_all = len(y)
    var_d = n_all / (n_all - 1) * float(np.sum(w**2 * z**2)) / (N_d**2)
    se = math.sqrt(max(0.0, var_d))

    z_crit = float(scipy_stats.norm.ppf(0.975))
    return {
        "mean": float(y_bar_d),
        "se": float(se),
        "ci_lower": float(y_bar_d - z_crit * se),
        "ci_upper": float(y_bar_d + z_crit * se),
        "n_domain": n_domain,
    }


# ===========================================================================
# SECTION 6 -- COMPLEX SURVEY GLM
# ===========================================================================


def complex_survey_glm(
    df: pd.DataFrame,
    formula: str,
    weight_col: str,
    *,
    family: str = "gaussian",
    cluster_col: str | None = None,
    strata_col: str | None = None,
) -> object:
    """
    Fit a GLM with complex survey design (weights, optional clustering, strata).

    The coefficients are the weighted IRLS fit and the standard errors the
    design-based linearisation variance of ``survey::svyglm``: score totals
    per PSU (``cluster_col``, or each unit), their between-PSU variance
    within strata (``strata_col``) scaled by n_h/(n_h - 1), and t tests on
    (PSUs - strata) + 1 - p degrees of freedom.

    Supported families (string): ``"gaussian"``, ``"binomial"``, ``"poisson"``,
    ``"gamma"``, ``"negativebinomial"``.

    :param df: Input DataFrame.
    :param formula: Patsy-compatible formula string (e.g. ``"y ~ x1 + x2"``).
    :param weight_col: Column name of analytic/probability survey weights.
    :param family: GLM family string. Default ``"gaussian"``.
    :param cluster_col: Column identifying clusters for robust SE. Default None.
    :param strata_col: Column identifying strata (informational only). Default None.
    :return: Fitted statsmodels GLM result object.
    :raises ValueError: If required columns are missing or family is unrecognised.

    References
    ----------
    Lumley, T. (2010). Complex Surveys: A Guide to Analysis Using R. Wiley.
    White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator
        and a direct test for heteroskedasticity. Econometrica, 48(4), 817-838.
    """
    if weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in DataFrame.")
    if cluster_col is not None and cluster_col not in df.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in DataFrame.")
    if strata_col is not None and strata_col not in df.columns:
        raise ValueError(f"Strata column '{strata_col}' not found in DataFrame.")

    family_str = family.lower().replace("-", "").replace("_", "")
    if family_str not in _glm_core.FAMILIES:
        raise ValueError(f"Unknown family '{family}'. Choose from: {list(_glm_core.FAMILIES)}.")
    w = df[weight_col].astype(float)
    if np.any(w.values <= 0):
        raise ValueError("All survey weights must be > 0.")
    return _svy_glm(
        formula,
        df,
        family_str,
        w,
        psu=None if cluster_col is None else df[cluster_col],
        strata=None if strata_col is None else df[strata_col],
    )
