"""Sensitivity analysis for causal inference assumptions.

Provides tools to assess the robustness of causal effect estimates to
unmeasured confounding, model specification, and other threats to
internal validity.  Implements Rosenbaum bounds, E-value, Ding-VanderWeele
bias formulas, tipping-point analysis, and specification curve analysis.

References
----------
* Rosenbaum (2002). *Observational Studies*, 2nd ed.
* VanderWeele & Ding (2017). Sensitivity analysis in observational research.
* Cinelli & Hazlett (2020). Making sense of sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn import _stats_core as stats

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class EValueResult:
    """Result from E-value computation."""

    point_estimate: float
    e_value_point: float
    e_value_ci: float
    rr: float
    ci_lower: float
    ci_upper: float
    interpretation: str


@dataclass
class RosenbaumBounds:
    """Result from Rosenbaum sensitivity analysis."""

    gamma_values: np.ndarray
    p_upper: np.ndarray
    p_lower: np.ndarray
    critical_gamma: float
    method: str
    interpretation: str


@dataclass
class TippingPointResult:
    """Result from tipping-point analysis."""

    delta_values: np.ndarray
    adjusted_estimates: np.ndarray
    adjusted_p_values: np.ndarray
    tipping_point: float
    original_estimate: float
    interpretation: str


@dataclass
class OmittedVariableBias:
    """Result from omitted variable bias analysis (Cinelli & Hazlett)."""

    estimate: float
    se: float
    rv_q: float  # robustness value at q
    rv_qa: float  # robustness value at q,alpha
    partial_r2_treatment: float
    benchmark_bounds: dict[str, tuple[float, float]]
    interpretation: str


@dataclass
class SpecificationCurveResult:
    """Result from specification curve analysis."""

    estimates: np.ndarray
    ses: np.ndarray
    p_values: np.ndarray
    specifications: list[dict]
    median_estimate: float
    iqr_lower: float
    iqr_upper: float
    pct_significant: float
    pct_same_sign: float


# ---------------------------------------------------------------------------
# E-value (VanderWeele & Ding 2017)
# ---------------------------------------------------------------------------


def _rr_to_evalue(rr: float) -> float:
    """Convert a risk ratio to its E-value."""
    if rr < 1:
        rr = 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


def e_value_rr(
    rr: float,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
) -> EValueResult:
    """Compute E-value for a risk ratio.

    The E-value is the minimum strength of association (on the risk ratio
    scale) that an unmeasured confounder would need to have with both
    treatment and outcome to fully explain away the observed association.

    Parameters
    ----------
    rr : float
        Observed risk ratio.
    ci_lower : float, optional
        Lower bound of 95% CI for the RR.
    ci_upper : float, optional
        Upper bound of 95% CI for the RR.

    Returns
    -------
    EValueResult
    """
    e_point = _rr_to_evalue(rr)

    if ci_lower is not None and ci_upper is not None:
        # E-value for the CI limit closest to 1.
        if rr >= 1:
            e_ci = _rr_to_evalue(ci_lower) if ci_lower > 1 else 1.0
        else:
            e_ci = _rr_to_evalue(ci_upper) if ci_upper < 1 else 1.0
    else:
        e_ci = float("nan")
        ci_lower = ci_lower or float("nan")
        ci_upper = ci_upper or float("nan")

    interpretation = (
        f"An unmeasured confounder would need RR >= {e_point:.2f} with both "
        f"treatment and outcome to explain away the point estimate (RR={rr:.2f}). "
        f"To move the CI to include the null, RR >= {e_ci:.2f} would be needed."
    )

    return EValueResult(
        point_estimate=rr,
        e_value_point=e_point,
        e_value_ci=e_ci,
        rr=rr,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        interpretation=interpretation,
    )


def e_value_or(
    odds_ratio: float,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    prevalence: float | None = None,
) -> EValueResult:
    """Compute E-value for an odds ratio.

    For rare outcomes (prevalence < 15%, or ``prevalence`` not given) the
    OR approximates the RR.  For common outcomes the OR is converted with
    :math:`RR \\approx \\sqrt{OR}` (VanderWeele & Ding 2017), as
    ``EValue::evalues.OR(rare = FALSE)``.

    Parameters
    ----------
    odds_ratio : float
        Observed odds ratio.
    ci_lower : float, optional
        Lower CI bound.
    ci_upper : float, optional
        Upper CI bound.
    prevalence : float, optional
        Outcome prevalence; 0.15 or more selects the common-outcome
        conversion.

    Returns
    -------
    EValueResult
    """
    if prevalence is not None and prevalence >= 0.15:
        rr = np.sqrt(odds_ratio)
        ci_lower = np.sqrt(ci_lower) if ci_lower is not None else None
        ci_upper = np.sqrt(ci_upper) if ci_upper is not None else None
    else:
        rr = odds_ratio
    return e_value_rr(rr, ci_lower, ci_upper)


def e_value_hr(
    hr: float,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    rare: bool = False,
) -> EValueResult:
    """Compute E-value for a hazard ratio.

    For a common outcome the HR is converted with VanderWeele's (2017)
    approximation :math:`RR = (1 - 0.5^{\\sqrt{HR}}) / (1 - 0.5^{\\sqrt{1/HR}})`;
    for a rare outcome (``rare=True``) the HR approximates the RR, as
    ``EValue::evalues.HR``.

    Parameters
    ----------
    hr : float
        Observed hazard ratio.
    ci_lower : float, optional
        Lower CI bound.
    ci_upper : float, optional
        Upper CI bound.
    rare : bool
        Outcome rare (< 15%) at the end of follow-up.

    Returns
    -------
    EValueResult
    """

    def to_rr(x):
        if x is None or x <= 0:
            return None
        if rare or x == 1:
            return x
        return (1 - 0.5 ** np.sqrt(x)) / (1 - 0.5 ** np.sqrt(1 / x))

    return e_value_rr(to_rr(hr), to_rr(ci_lower), to_rr(ci_upper))


def e_value_d(
    d: float,
    se: float | None = None,
    n: int | None = None,
) -> EValueResult:
    """Compute E-value for a standardized mean difference (Cohen's d).

    Converts d to a risk ratio with the VanderWeele & Ding (2017)
    approximation :math:`RR \\approx \\exp(0.91 d)`, CI
    :math:`\\exp(0.91 d \\pm 1.78\\,s)`, as ``EValue::evalues.MD``.

    Parameters
    ----------
    d : float
        Standardized mean difference.
    se : float, optional
        Standard error of d.
    n : int, optional
        Sample size (used to compute SE if not provided).

    Returns
    -------
    EValueResult
    """
    rr = np.exp(0.91 * d)
    if se is None and n is not None:
        se = np.sqrt(4 / n)
    if se is not None:
        rr_lo = np.exp(0.91 * d - 1.78 * se)
        rr_hi = np.exp(0.91 * d + 1.78 * se)
    else:
        rr_lo = None
        rr_hi = None
    return e_value_rr(rr, rr_lo, rr_hi)


# ---------------------------------------------------------------------------
# Rosenbaum bounds
# ---------------------------------------------------------------------------


def rosenbaum_bounds(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    gamma_range: np.ndarray | None = None,
    method: str = "wilcoxon",
) -> RosenbaumBounds:
    """Rosenbaum sensitivity analysis for matched pair designs.

    Tests how strong hidden bias (Gamma) would need to be to alter
    the study's conclusions.

    Parameters
    ----------
    treated_outcomes : array-like
        Outcomes for treated units in matched pairs.
    control_outcomes : array-like
        Outcomes for control units in matched pairs.
    gamma_range : array-like, optional
        Range of Gamma values to test.  Default: 1.0 to 5.0 by 0.25.
    method : str
        Test method: 'wilcoxon' (default), 'sign', 'mcnemar'.

    Returns
    -------
    RosenbaumBounds
    """
    t = np.asarray(treated_outcomes, dtype=float)
    c = np.asarray(control_outcomes, dtype=float)
    n = len(t)
    diffs = t - c

    if gamma_range is None:
        gamma_range = np.arange(1.0, 5.25, 0.25)

    gamma_range = np.asarray(gamma_range, dtype=float)
    p_upper = np.empty(len(gamma_range))
    p_lower = np.empty(len(gamma_range))

    if method == "wilcoxon":
        # zero differences carry no sign information (Wilcoxon's rule)
        diffs = diffs[diffs != 0]
        ranks = stats.rankdata(np.abs(diffs))
        signs = np.sign(diffs)
        t_obs = np.sum(ranks[signs > 0])

        for i, gamma in enumerate(gamma_range):
            # Under Gamma-bias, the probability that each pair is treated
            # is between 1/(1+Gamma) and Gamma/(1+Gamma).
            p_treat = gamma / (1 + gamma)

            # Upper bound: maximize p-value.
            expected_upper = np.sum(ranks * p_treat)
            var_upper = np.sum(ranks**2 * p_treat * (1 - p_treat))
            z_upper = (t_obs - expected_upper) / np.sqrt(max(var_upper, 1e-10))
            p_upper[i] = stats.norm.sf(z_upper)

            # Lower bound.
            p_treat_low = 1 / (1 + gamma)
            expected_lower = np.sum(ranks * p_treat_low)
            var_lower = np.sum(ranks**2 * p_treat_low * (1 - p_treat_low))
            z_lower = (t_obs - expected_lower) / np.sqrt(max(var_lower, 1e-10))
            p_lower[i] = stats.norm.sf(z_lower)

    elif method == "sign":
        # zero differences carry no sign; n counts the non-zero pairs
        n_positive = np.sum(diffs > 0)
        n = int(np.sum(diffs != 0))

        for i, gamma in enumerate(gamma_range):
            p_treat = gamma / (1 + gamma)
            # Upper bound: binomial test.
            p_upper[i] = stats.binom.sf(n_positive - 1, n, p_treat)
            p_treat_low = 1 / (1 + gamma)
            p_lower[i] = stats.binom.sf(n_positive - 1, n, p_treat_low)

    elif method == "mcnemar":
        # For binary outcomes: discordant pairs.
        b = np.sum((t == 1) & (c == 0))  # treated=1, control=0
        cc = np.sum((t == 0) & (c == 1))  # treated=0, control=1
        n_disc = b + cc

        for i, gamma in enumerate(gamma_range):
            p_treat = gamma / (1 + gamma)
            p_upper[i] = stats.binom.sf(int(b) - 1, int(n_disc), p_treat)
            p_treat_low = 1 / (1 + gamma)
            p_lower[i] = stats.binom.sf(int(b) - 1, int(n_disc), p_treat_low)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Critical Gamma: smallest Gamma where upper bound p > 0.05.
    critical_idx = np.where(p_upper > 0.05)[0]
    critical_gamma = float(gamma_range[critical_idx[0]]) if len(critical_idx) > 0 else float(gamma_range[-1])

    interpretation = (
        f"The study conclusion is sensitive to hidden bias at Gamma = {critical_gamma:.2f}. "
        f"An unobserved covariate that changes the odds of treatment by a factor of "
        f"{critical_gamma:.2f} could explain away the result."
    )

    return RosenbaumBounds(
        gamma_values=gamma_range,
        p_upper=p_upper,
        p_lower=p_lower,
        critical_gamma=critical_gamma,
        method=method,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Tipping point analysis
# ---------------------------------------------------------------------------


def tipping_point_analysis(
    estimate: float,
    se: float,
    n_treated: int,
    n_control: int,
    delta_range: np.ndarray | None = None,
    outcome_type: str = "continuous",
) -> TippingPointResult:
    """Tipping-point analysis for missing data sensitivity.

    Evaluates how much the treatment effect changes if missing outcomes
    are systematically different from observed outcomes, shifting the
    estimate by :math:`\\delta`.  The tipping point is the smallest shift
    that makes the two-sided 5% test non-significant,
    :math:`\\delta^* = \\hat\\tau - \\mathrm{sign}(\\hat\\tau)\\, z_{0.975}\\, se`
    (0 when the estimate is already non-significant); the grid reports
    the adjusted estimates and p-values over ``delta_range``.

    Parameters
    ----------
    estimate : float
        Observed treatment effect estimate.
    se : float
        Standard error of the estimate.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    delta_range : array-like, optional
        Range of bias parameters to evaluate.
    outcome_type : str
        'continuous' or 'binary'.

    Returns
    -------
    TippingPointResult
    """
    if outcome_type not in ("continuous", "binary"):
        raise ValueError(f"outcome_type must be 'continuous' or 'binary' (got {outcome_type!r})")
    if delta_range is None:
        max_delta = abs(estimate) * 3
        if outcome_type == "binary":
            max_delta = min(max_delta, 1.0)
        delta_range = np.linspace(-max_delta, max_delta, 101)
    delta_range = np.asarray(delta_range, dtype=float)
    adjusted_estimates = estimate - delta_range
    adjusted_p = 2 * stats.norm.sf(np.abs(adjusted_estimates / se))
    z = stats.norm.ppf(0.975)
    tipping_point = float(estimate - np.sign(estimate) * z * se) if abs(estimate) > z * se else 0.0
    _robust = abs(tipping_point) > abs(estimate)
    _robustness_msg = (
        "This suggests the result is robust."
        if _robust
        else "This suggests the result may be sensitive to missing data."
    )
    interpretation = (
        f"The observed estimate ({estimate:.4f}) becomes non-significant when outcomes for "
        f"missing data differ by delta = {tipping_point:.4f}. {_robustness_msg}"
    )
    return TippingPointResult(
        delta_values=delta_range,
        adjusted_estimates=adjusted_estimates,
        adjusted_p_values=adjusted_p,
        tipping_point=tipping_point,
        original_estimate=estimate,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Omitted variable bias (Cinelli & Hazlett 2020)
# ---------------------------------------------------------------------------


def omitted_variable_bias(
    estimate: float,
    se: float,
    dof: int,
    r2_yd_x: float,
    partial_r2_treatment: float,
    q: float = 1.0,
    alpha: float = 0.05,
    benchmark_covariates: dict[str, float | tuple[float, float]] | None = None,
    kd: float = 1.0,
    ky: float | None = None,
) -> OmittedVariableBias:
    """Omitted-variable-bias sensitivity (Cinelli & Hazlett 2020).

    Robustness values as ``sensemakr::robustness_value``: with
    :math:`f_q = q |t| / \\sqrt{dof}`, :math:`RV_q =
    \\tfrac12(\\sqrt{f_q^4 + 4 f_q^2} - f_q^2)`, and :math:`RV_{q,\\alpha}`
    the same in :math:`f_q - f^*` with :math:`f^* = |t^*_{\\alpha, dof-1}| /
    \\sqrt{dof - 1}` (the extreme robustness value when :math:`f_q > 1/f^*`).
    Benchmark bounds follow ``sensemakr::ovb_bounds``: a confounder
    ``kd`` / ``ky`` times as strong as the benchmark covariate.

    Parameters
    ----------
    estimate : float
        Treatment coefficient estimate.
    se : float
        Standard error of the estimate.
    dof : int
        Residual degrees of freedom.
    r2_yd_x : float
        Partial R-squared of treatment with outcome (controlling for X).
    partial_r2_treatment : float
        Same as r2_yd_x (for clarity).
    q : float
        Fraction of the estimate to be explained away.  Default 1.0 (full).
    alpha : float
        Significance level.
    benchmark_covariates : dict, optional
        Covariate name to ``(r2_dxj_x, r2_yxj_dx)``: its partial R-squared
        with the treatment (given the other covariates) and with the
        outcome (given treatment and the other covariates).  A single
        number is used for both.
    kd, ky : float
        Strength multipliers of the confounder relative to a benchmark
        (``ky`` defaults to ``kd``).

    Returns
    -------
    OmittedVariableBias
        ``benchmark_bounds`` maps each covariate to a dict with
        ``r2dz_x``, ``r2yz_dx``, ``adjusted_estimate``, ``adjusted_se``,
        ``adjusted_lower_ci`` and ``adjusted_upper_ci``.
    """
    t_stat = estimate / se

    def _rv(alpha_):
        fq = q * abs(t_stat) / np.sqrt(dof)
        f_crit = abs(stats.t.ppf(alpha_ / 2, dof - 1)) / np.sqrt(dof - 1) if alpha_ < 1 else 0.0
        fqa = fq - f_crit
        if fqa <= 0:
            return 0.0
        if f_crit > 0 and fq > 1 / f_crit:
            return float((fq**2 - f_crit**2) / (1 + fq**2))
        return float(2 / (1 + np.sqrt(1 + 4 / fqa**2)))

    rv_q = _rv(1.0)
    rv_qa = _rv(alpha)
    ky = kd if ky is None else ky
    bounds = {}
    for name, r2 in (benchmark_covariates or {}).items():
        r2dxj, r2yxj = (r2, r2) if isinstance(r2, (int, float)) else r2
        r2dz = kd * r2dxj / (1 - r2dxj)
        if r2dz >= 1:
            raise ValueError(f"Implied bound on r2dz.x >= 1 for {name!r}; use a lower kd.")
        r2zxj = kd * r2dxj**2 / ((1 - kd * r2dxj) * (1 - r2dxj))
        if r2zxj >= 1:
            raise ValueError(f"Impossible kd value for {name!r}; use a lower kd.")
        r2yz = min(((np.sqrt(ky) + np.sqrt(r2zxj)) / np.sqrt(1 - r2zxj)) ** 2 * (r2yxj / (1 - r2yxj)), 1.0)
        bias = np.sqrt(r2yz * r2dz / (1 - r2dz)) * se * np.sqrt(dof)
        adj = float(np.sign(estimate) * (abs(estimate) - bias))
        adj_se = float(np.sqrt((1 - r2yz) / (1 - r2dz)) * se * np.sqrt(dof / (dof - 1)))
        tc = stats.t.ppf(1 - alpha / 2, dof)
        bounds[name] = {
            "r2dz_x": float(r2dz),
            "r2yz_dx": float(r2yz),
            "adjusted_estimate": adj,
            "adjusted_se": adj_se,
            "adjusted_lower_ci": adj - tc * adj_se,
            "adjusted_upper_ci": adj + tc * adj_se,
        }
    interpretation = (
        f"To explain away {q * 100:.0f}% of the estimate ({estimate:.4f}), an unobserved "
        f"confounder would need partial R2 >= {rv_q:.4f} with both treatment and outcome. "
        f"To make the CI include zero, partial R2 >= {rv_qa:.4f}."
    )
    return OmittedVariableBias(
        estimate=estimate,
        se=se,
        rv_q=rv_q,
        rv_qa=rv_qa,
        partial_r2_treatment=partial_r2_treatment,
        benchmark_bounds=bounds,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Specification curve analysis
# ---------------------------------------------------------------------------


def specification_curve(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariate_sets: list[list[str]],
    sample_filters: list[tuple[str, callable]] | None = None,
    model_types: list[str] | None = None,
    alpha: float = 0.05,
) -> SpecificationCurveResult:
    """Run a specification curve analysis.

    Estimates the treatment effect across many reasonable model specifications
    to assess robustness.

    Parameters
    ----------
    data : pd.DataFrame
        Analysis dataset.
    outcome : str
        Outcome variable name.
    treatment : str
        Treatment variable name.
    covariate_sets : list of list[str]
        Different sets of covariates to try.
    sample_filters : list of (name, callable), optional
        Different sample restrictions to try.
    model_types : list[str], optional
        Model types: 'ols', 'logistic', 'robust'.  Default: ['ols'].
    alpha : float
        Significance level.

    Returns
    -------
    SpecificationCurveResult
    """
    from morie.fn import _glm_core as sm

    if model_types is None:
        model_types = ["ols"]
    if sample_filters is None:
        sample_filters = [("full_sample", lambda df: df)]

    estimates = []
    ses = []
    p_values = []
    specifications = []

    for filter_name, filter_fn in sample_filters:
        filtered = filter_fn(data).copy()
        if len(filtered) < 10:
            continue

        for cov_set in covariate_sets:
            # Check all covariates exist.
            missing_cols = [c for c in cov_set if c not in filtered.columns]
            if missing_cols:
                continue

            for model_type in model_types:
                try:
                    y = filtered[outcome].astype(float)
                    X_vars = [treatment] + cov_set
                    X = filtered[X_vars].astype(float)
                    X = sm.add_constant(X, has_constant="add")

                    # Drop any rows with NaN.
                    mask = y.notna() & X.notna().all(axis=1)
                    y = y[mask]
                    X = X.loc[mask]

                    if len(y) < len(X_vars) + 2:
                        continue

                    if model_type == "ols":
                        model = sm.OLS(y, X).fit()
                    elif model_type == "logistic":
                        model = sm.Logit(y, X).fit(disp=0)
                    elif model_type == "robust":
                        model = sm.RLM(y, X).fit()
                    else:
                        continue

                    est = model.params[treatment]
                    se = model.bse[treatment]
                    pval = model.pvalues[treatment]

                    estimates.append(est)
                    ses.append(se)
                    p_values.append(pval)
                    specifications.append(
                        {
                            "sample": filter_name,
                            "covariates": cov_set,
                            "model": model_type,
                            "n": int(len(y)),
                            "estimate": est,
                            "se": se,
                            "p_value": pval,
                        }
                    )

                except Exception:
                    continue

    if not estimates:
        return SpecificationCurveResult(
            estimates=np.array([]),
            ses=np.array([]),
            p_values=np.array([]),
            specifications=[],
            median_estimate=float("nan"),
            iqr_lower=float("nan"),
            iqr_upper=float("nan"),
            pct_significant=0.0,
            pct_same_sign=0.0,
        )

    est_arr = np.array(estimates)
    se_arr = np.array(ses)
    p_arr = np.array(p_values)

    median_est = float(np.median(est_arr))
    q25, q75 = np.percentile(est_arr, [25, 75])

    n_sig = np.sum(p_arr <= alpha)
    modal_sign = np.sign(median_est)
    n_same_sign = np.sum(np.sign(est_arr) == modal_sign)

    return SpecificationCurveResult(
        estimates=est_arr,
        ses=se_arr,
        p_values=p_arr,
        specifications=specifications,
        median_estimate=median_est,
        iqr_lower=float(q25),
        iqr_upper=float(q75),
        pct_significant=float(n_sig / len(est_arr) * 100),
        pct_same_sign=float(n_same_sign / len(est_arr) * 100),
    )


# ---------------------------------------------------------------------------
# Manski bounds (partial identification)
# ---------------------------------------------------------------------------


def manski_bounds(
    outcome_treated: np.ndarray,
    outcome_control: np.ndarray,
    p_treated: float,
    outcome_range: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Compute Manski worst-case bounds for the ATE.

    Under no assumptions about selection, the ATE is only partially
    identified (Manski 1990): with the outcome in :math:`[a, b]`,
    :math:`E[Y_1] \\in [p\\bar y_1 + a(1-p),\\; p\\bar y_1 + b(1-p)]` and
    :math:`E[Y_0] \\in [(1-p)\\bar y_0 + a p,\\; (1-p)\\bar y_0 + b p]`, so the
    bounds always have width :math:`b - a`.

    Parameters
    ----------
    outcome_treated : array-like
        Outcomes for treated units.
    outcome_control : array-like
        Outcomes for control units.
    p_treated : float
        Proportion treated.
    outcome_range : tuple[float, float], optional
        Logical bounds on the outcome.  Default: (0, 1) for binary.

    Returns
    -------
    dict
        With keys: lower_bound, upper_bound, point_estimate, width.
    """
    y1 = np.asarray(outcome_treated, dtype=float)
    y0 = np.asarray(outcome_control, dtype=float)
    if outcome_range is None:
        outcome_range = (0.0, 1.0)
    y_min, y_max = outcome_range
    e1 = float(np.mean(y1))
    e0 = float(np.mean(y0))
    p1 = p_treated
    p0 = 1 - p_treated
    lo = e1 * p1 + y_min * p0 - (e0 * p0 + y_max * p1)
    hi = e1 * p1 + y_max * p0 - (e0 * p0 + y_min * p1)
    return {"lower_bound": float(lo), "upper_bound": float(hi), "point_estimate": e1 - e0, "width": float(hi - lo)}


# ---------------------------------------------------------------------------
# Bias-adjusted estimates
# ---------------------------------------------------------------------------


def bias_adjusted_estimate(
    estimate: float,
    se: float,
    rr_ud: float,
    rr_eu: float,
    prevalence_confounder: float | None = None,
) -> dict[str, float]:
    """Compute bias-adjusted treatment effect on the log-RR scale.

    Without ``prevalence_confounder`` the adjustment is the Ding &
    VanderWeele (2016) bounding factor
    :math:`B = RR_{UD} RR_{EU} / (RR_{UD} + RR_{EU} - 1)`, the largest
    bias any confounder with those strengths can produce, applied toward
    the null.  With the prevalence :math:`p_0` of a binary confounder
    among the unexposed, ``rr_eu`` is its prevalence ratio
    (:math:`p_1 = RR_{EU} p_0`) and the bias is Schlesselman's (1978)
    exact factor :math:`(1 + (RR_{UD}-1)p_1) / (1 + (RR_{UD}-1)p_0)`.

    Parameters
    ----------
    estimate : float
        Observed treatment effect (log-RR scale).
    se : float
        Standard error.
    rr_ud : float
        Risk ratio relating confounder to outcome.
    rr_eu : float
        Risk ratio relating treatment to confounder.
    prevalence_confounder : float, optional
        Prevalence of the confounder among the unexposed.

    Returns
    -------
    dict
        With keys: adjusted_estimate, bias, adjusted_ci_lower,
        adjusted_ci_upper, original_estimate (``bias`` on the log scale).
    """
    if prevalence_confounder is None:
        bias = float(np.log(rr_ud * rr_eu / (rr_ud + rr_eu - 1))) * (1.0 if estimate >= 0 else -1.0)
    else:
        p0 = float(prevalence_confounder)
        p1 = rr_eu * p0
        if not 0 <= p1 <= 1:
            raise ValueError("rr_eu * prevalence_confounder must be a probability.")
        bias = float(np.log((1 + (rr_ud - 1) * p1) / (1 + (rr_ud - 1) * p0)))
    adjusted = estimate - bias
    z = stats.norm.ppf(0.975)
    return {
        "adjusted_estimate": float(adjusted),
        "bias": bias,
        "adjusted_ci_lower": float(adjusted - z * se),
        "adjusted_ci_upper": float(adjusted + z * se),
        "original_estimate": float(estimate),
    }


# ---------------------------------------------------------------------------
# Quantitative bias analysis (probabilistic)
# ---------------------------------------------------------------------------


def probabilistic_bias_analysis(
    estimate: float,
    se: float,
    n_simulations: int = 10000,
    bias_parms: dict[str, tuple[float, float]] | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Probabilistic (Monte Carlo) sensitivity analysis.

    Draws bias parameters from specified prior distributions and the
    estimate from its sampling distribution, and removes Schlesselman's
    binary-confounder bias :math:`\\log[(1 + (RR_{UD}-1)p_1) /
    (1 + (RR_{UD}-1)p_0)]` with :math:`p_0` the drawn prevalence among
    the unexposed and :math:`p_1 = \\min(RR_{EU} p_0, 1)` (Lash, Fox &
    Fink 2009, ch. 8).

    Parameters
    ----------
    estimate : float
        Observed estimate (log-RR scale).
    se : float
        Standard error.
    n_simulations : int
        Number of Monte Carlo draws.
    bias_parms : dict, optional
        Parameters as (mean, sd) tuples for: 'rr_ud', 'rr_eu', 'prevalence'.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Summary of bias-adjusted estimate distribution.
    """
    rng = np.random.default_rng(seed)
    if bias_parms is None:
        bias_parms = {"rr_ud": (1.5, 0.3), "rr_eu": (1.5, 0.3), "prevalence": (0.3, 0.1)}
    rr_ud = np.abs(rng.normal(bias_parms["rr_ud"][0], bias_parms["rr_ud"][1], n_simulations))
    rr_eu = np.abs(rng.normal(bias_parms["rr_eu"][0], bias_parms["rr_eu"][1], n_simulations))
    p0 = np.clip(rng.normal(bias_parms["prevalence"][0], bias_parms["prevalence"][1], n_simulations), 0.01, 0.99)
    estimates_with_error = rng.normal(estimate, se, n_simulations)
    p1 = np.minimum(rr_eu * p0, 1.0)
    biases = np.log((1 + (rr_ud - 1) * p1) / (1 + (rr_ud - 1) * p0))
    adjusted = estimates_with_error - biases
    return {
        "original_estimate": float(estimate),
        "median_adjusted": float(np.median(adjusted)),
        "mean_adjusted": float(np.mean(adjusted)),
        "ci_2.5": float(np.percentile(adjusted, 2.5)),
        "ci_97.5": float(np.percentile(adjusted, 97.5)),
        "pct_null_included": float(np.mean((adjusted < 0) != (estimate < 0)) * 100),
        "pct_same_sign": float(np.mean(np.sign(adjusted) == np.sign(estimate)) * 100),
        "n_simulations": n_simulations,
    }


# ---------------------------------------------------------------------------
# Cross-validation of sensitivity


def konfound(
    estimate: float,
    se: float,
    n: int,
    n_covariates: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Robustness of an inference to replacement and confounding (Frank et al.).

    Native ``konfound::pkonfound`` for a regression coefficient (two
    tails, null 0): the threshold is :math:`t^* se` with :math:`t^*` on
    :math:`n - n_{cov} - 2` df; the percent bias to invalidate is
    :math:`100 (1 - t^* se / \\hat\\beta)` (to sustain when not
    significant) and RIR the corresponding number of cases; the impact
    threshold of a confounding variable is
    :math:`(r - r^*) / (1 \\pm |r^*|)` with :math:`r = t / \\sqrt{t^2 + df}`.
    Mirrors ``morie_sensitivity_konfound`` in the R arm.

    Parameters
    ----------
    estimate, se : float
        Coefficient and its standard error.
    n : int
        Number of observations.
    n_covariates : int
        Number of covariates besides the focal predictor.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        ``percent_bias_to_invalidate``, ``rir`` (cases to replace),
        ``impact_threshold_confounder``, ``beta_threshold``.
    """
    df = n - n_covariates - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df) * (-1 if estimate < 0 else 1)
    thr = t_crit * se
    # percent bias to invalidate, or to sustain when not significant
    pct = 100 * (1 - thr / estimate) if abs(estimate) > abs(thr) else 100 * (1 - estimate / thr)
    act_t = estimate / se
    act_r = act_t / np.sqrt(act_t**2 + df)
    crit_r = t_crit / np.sqrt(t_crit**2 + df)
    mp = 1 if -abs(thr) < estimate < abs(thr) else -1
    sign = 1 if estimate > thr else (-1 if estimate < thr else 0)
    itcv = sign * abs(act_r - crit_r) / (1 + mp * abs(crit_r))
    return {
        "percent_bias_to_invalidate": float(pct),
        "rir": int(round(n * pct / 100)),
        "impact_threshold_confounder": float(itcv),
        "beta_threshold": float(thr),
    }


def tipping_point_smd(effect_observed: float, smd: float) -> dict[str, float]:
    """Confounder-outcome effect that tips a ratio estimate to the null.

    Native ``tipr::tip``: an unmeasured confounder whose standardized
    mean difference between exposure groups is ``smd`` tips an observed
    ratio :math:`b` to 1 when its effect on the outcome is
    :math:`b^{1/smd}` (Lin, Psaty & Kronmal 1998).  Mirrors
    ``morie_sensitivity_tipping_point`` in the R arm.

    Parameters
    ----------
    effect_observed : float
        Observed ratio (e.g. risk ratio), the bound nearest the null.
    smd : float
        Exposure-confounder standardized mean difference.

    Returns
    -------
    dict
        ``effect_adjusted`` (1), ``confounder_outcome_effect``.
    """
    return {"effect_adjusted": 1.0, "confounder_outcome_effect": float(effect_observed ** (1 / smd))}


# ---------------------------------------------------------------------------


def sensitivity_summary(
    estimate: float,
    se: float,
    rr: float | None = None,
    odds_ratio: float | None = None,
    hazard_ratio: float | None = None,
    prevalence: float | None = None,
) -> pd.DataFrame:
    """Generate a comprehensive sensitivity analysis summary.

    Computes E-values, tipping points, and robustness metrics for
    a single treatment effect estimate.

    Parameters
    ----------
    estimate : float
        Treatment effect estimate.
    se : float
        Standard error.
    rr : float, optional
        Risk ratio (if applicable).
    odds_ratio : float, optional
        Odds ratio (if applicable).
    hazard_ratio : float, optional
        Hazard ratio (if applicable).
    prevalence : float, optional
        Outcome prevalence.

    Returns
    -------
    pd.DataFrame
        Summary table of sensitivity metrics.
    """
    rows = []

    # Basic info.
    ci_lo = estimate - 1.96 * se
    ci_hi = estimate + 1.96 * se
    z = estimate / se
    p = 2 * (stats.norm.sf(abs(z)))
    rows.append({"metric": "estimate", "value": estimate})
    rows.append({"metric": "se", "value": se})
    rows.append({"metric": "ci_lower", "value": ci_lo})
    rows.append({"metric": "ci_upper", "value": ci_hi})
    rows.append({"metric": "p_value", "value": p})

    # E-values.
    if rr is not None:
        ev = e_value_rr(rr, ci_lo if rr >= 1 else None, ci_hi if rr >= 1 else None)
        rows.append({"metric": "e_value_point", "value": ev.e_value_point})
        rows.append({"metric": "e_value_ci", "value": ev.e_value_ci})

    if odds_ratio is not None:
        ev = e_value_or(odds_ratio, prevalence=prevalence)
        rows.append({"metric": "e_value_or_point", "value": ev.e_value_point})
        rows.append({"metric": "e_value_or_ci", "value": ev.e_value_ci})

    if hazard_ratio is not None:
        ev = e_value_hr(hazard_ratio)
        rows.append({"metric": "e_value_hr_point", "value": ev.e_value_point})
        rows.append({"metric": "e_value_hr_ci", "value": ev.e_value_ci})

    # Tipping point.
    tp = tipping_point_analysis(estimate, se, 100, 100)
    rows.append({"metric": "tipping_point_delta", "value": tp.tipping_point})

    return pd.DataFrame(rows)
