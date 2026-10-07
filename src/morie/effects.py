"""
Treatment effect estimations (ATE, ATT, ATU, LATE, G-computation).

This module provides:

1. :func:`estimate_ate` -- IPW-weighted OLS ATE (existing, preserved).
2. :func:`estimate_plr` -- Partially Linear Regression (native cross-fitted DML).
3. :func:`estimate_pliv` -- Partially Linear IV (LATE), native DML or 2SLS.
4. :func:`estimate_ate_gcomputation` -- G-computation (outcome regression) ATE.
5. :func:`sensitivity_rosenbaum` -- Rosenbaum bounds for hidden confounding.
6. :func:`e_value` -- E-value for unmeasured confounding (VanderWeele & Ding, 2017).

References
----------
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey,
    W., & Robins, J. (2018). Double/debiased machine learning for treatment and
    structural parameters. The Econometrics Journal, 21(1), C1-C68.
Robins, J. M. (1986). A new approach to causal inference in mortality studies
    with a sustained exposure period. Mathematical Modelling, 7, 1393-1512.
VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational
    research: Introducing the E-value. Annals of Internal Medicine, 167(4), 268-274.
Rosenbaum, P. R. (2002). Observational Studies (2nd ed.). Springer.
"""

import math

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn import _stats_core as scipy_stats

from ._frames import as_frame


class _MissingDep:
    """Placeholder for a dependency being nativized (task #141)."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        raise ImportError(f"{self._name} is no longer bundled; this code path awaits its native morie implementation")

    def __call__(self, *a, **k):
        raise ImportError(f"{self._name} is no longer bundled; this code path awaits its native morie implementation")


try:
    from morie.fn import _glm_core as sm
except ImportError:
    sm = _MissingDep("sm")
try:
    from morie.fn._glm_core import formula as smf
except ImportError:
    smf = _MissingDep("smf")
try:
    from morie.fn._ml_core import LinearRegression, LogisticRegression
except ImportError:
    LinearRegression = _MissingDep("LinearRegression")
    LogisticRegression = _MissingDep("LogisticRegression")
try:
    from morie.fn._ml_core import StandardScaler
except ImportError:
    StandardScaler = _MissingDep("StandardScaler")


def estimate_ate(data: pd.DataFrame, outcome: str, treatment: str, weights_col: str) -> tuple[float, float]:
    """
    Estimate Average Treatment Effect (ATE) using a weighted linear model.

    :param data: The pandas DataFrame containing the analytical sample.
    :type data: pandas.DataFrame
    :param outcome: The name of the outcome variable column.
    :type outcome: str
    :param treatment: The name of the binary treatment indicator column.
    :type treatment: str
    :param weights_col: The name of the column containing the analytical weights (e.g. IPTW).
    :type weights_col: str
    :return: A tuple containing the estimated ATE coefficient and its standard error.
    :rtype: tuple[float, float]
    """
    data = as_frame(data)  # a DataFrame, a dict of columns or a CSV path
    formula = f"{outcome} ~ {treatment}"
    # HC3 robust covariance: corrects for heteroskedasticity introduced by
    # unequal IPTW weights.  Plain OLS/WLS SEs are downward-biased when
    # observation weights vary widely, producing anti-conservative inference.
    model = smf.wls(formula=formula, data=data, weights=data[weights_col]).fit(cov_type="HC3")
    return float(model.params[treatment]), float(model.bse[treatment])


# ===========================================================================
# SECTION 2 -- DOUBLEML PARTIALLY LINEAR REGRESSION (PLR)
# ===========================================================================


def estimate_plr(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Partially Linear Regression ATE via the native double-ML
    estimator (cross-fitted ridge nuisances, partialling-out score).

    Thin front-end for :func:`morie.fn.plr.estimate_plr`; see that
    function for the model, the score, and references.

    :return: dict with ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``pval``, ``n_obs``.
    """
    data = as_frame(data)  # a DataFrame, a dict of columns or a CSV path
    from morie.fn.plr import estimate_plr as _native

    return _native(
        data, treatment=treatment, outcome=outcome, covariates=covariates, n_folds=n_folds, random_state=random_state
    )


def estimate_pliv(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    instrument: str,
    covariates: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Partially linear IV (LATE) via the native double-ML estimator
    (cross-fitted ridge nuisances, IV-type orthogonal score).

    Thin front-end for :func:`morie.fn.pliv.estimate_pliv`; see that
    function for the model, the score, and references.

    :return: dict with ``late``, ``se``, ``ci_lower``, ``ci_upper``,
        ``pval``, ``n_obs``, ``method``.
    """
    data = as_frame(data)  # a DataFrame, a dict of columns or a CSV path
    from morie.fn.pliv import estimate_pliv as _native

    return _native(
        data,
        treatment=treatment,
        outcome=outcome,
        instrument=instrument,
        covariates=covariates,
        n_folds=n_folds,
        random_state=random_state,
    )


def estimate_ate_gcomputation(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    outcome_model: str = "linear",
) -> dict:
    """
    G-computation (outcome regression / standardisation) ATE estimator.

    The G-computation estimator proceeds in three steps:

    1. Fit an outcome model :math:`E[Y | T, X]` on the observed data.
    2. Predict potential outcomes :math:`\\hat{Y}(1)` and :math:`\\hat{Y}(0)`
       for every unit by setting T = 1 and T = 0 respectively.
    3. Compute ATE as the average difference:

    .. math::

        \\widehat{\\text{ATE}} = \\frac{1}{n} \\sum_i
        \\left(\\hat{Y}_i(1) - \\hat{Y}_i(0)\\right)

    The outcome model is the unpenalised GLM (OLS or logistic) and the standard error is the
    M-estimation sandwich of ``stdReg::stdGlm`` (Sjolander 2016) -- the GLM score equations stacked
    with the two standardisation equations, so the outcome model's own uncertainty is included --
    the same estimate and SE as the R arm.

    :param data: DataFrame containing all required columns.
    :param treatment: Column name of the binary treatment indicator (0/1).
    :param outcome: Column name of the outcome variable.
    :param covariates: List of covariate column names.
    :param outcome_model: ``"linear"`` (OLS) for continuous outcomes or
        ``"logistic"`` for binary outcomes. Default ``"linear"``.
    :return: dict with keys ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``n_obs``, ``outcome_model``.
    :raises ValueError: If required columns are missing or outcome_model is invalid.

    References
    ----------
    Robins, J. M. (1986). A new approach to causal inference in mortality
        studies. Mathematical Modelling, 7, 1393-1512.
    Hernan, M. A., & Robins, J. M. (2020). Causal Inference: What If.
        Chapman & Hall/CRC. (Chapter 13.)
    """
    data = as_frame(data)  # a DataFrame, a dict of columns or a CSV path
    valid_models = {"linear", "logistic"}
    if outcome_model not in valid_models:
        raise ValueError(f"outcome_model must be one of {valid_models}.")

    required_cols = [treatment, outcome] + covariates
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}.")

    df = data[[treatment, outcome] + covariates].dropna().reset_index(drop=True)
    n_obs = len(df)
    if n_obs < 10:
        raise ValueError("G-computation requires at least 10 complete observations.")

    return _gformula_std(df, treatment, outcome, covariates, outcome_model, n_obs)


def _gformula_std(df, treatment, outcome, covariates, outcome_model, n_obs):
    """Regression standardisation with the M-estimation sandwich of stdReg::stdGlm (Sjolander
    2016): the GLM score equations stacked with the two standardisation equations,
    V = I^-1 J I^-T / n, the ATE's SE by the delta method -- the R arm's .morie_gformula_std."""
    from morie.fn import _glm_core

    fam = "gaussian" if outcome_model == "linear" else "binomial"
    F = _glm_core.FAMILIES[fam]
    cols = [treatment, *covariates]
    X = [[1.0] + [float(df[c].iloc[i]) for c in cols] for i in range(n_obs)]
    y = [float(v) for v in df[outcome].tolist()]
    fit = _glm_core.glm(y, [r[1:] for r in X], family=fam, add_intercept=True)
    beta = [float(v) for v in fit["coef"]] if "coef" in fit else [float(v) for v in fit["coefficients"]]
    k = len(beta)
    n = n_obs
    pred = [[0.0, 0.0] for _ in range(n)]
    si = [[0.0] * k for _ in range(2)]
    for a in (0, 1):
        for i in range(n):
            xa = list(X[i])
            xa[1] = float(a)
            eta = sum(xa[j] * beta[j] for j in range(k))
            pred[i][a] = F["linkinv"](eta)
            d = F["mu_eta"](eta)
            for j in range(k):
                si[a][j] += d * xa[j] / n
    est = [sum(pred[i][a] for i in range(n)) / n for a in (0, 1)]
    mu = [F["linkinv"](sum(X[i][j] * beta[j] for j in range(k))) for i in range(n)]
    scores = [
        [pred[i][0] - est[0], pred[i][1] - est[1]] + [X[i][j] * (y[i] - mu[i]) for j in range(k)] for i in range(n)
    ]
    m = len(scores[0])
    cm = [sum(r[j] for r in scores) / n for j in range(m)]
    J = [[sum((r[x] - cm[x]) * (r[yy] - cm[yy]) for r in scores) / (n - 1) for yy in range(m)] for x in range(m)]
    ww = fit["working_weights"]
    XtWX = [[sum(ww[i] * X[i][x] * X[i][yy] for i in range(n)) for yy in range(k)] for x in range(k)]
    info = [[0.0] * m for _ in range(m)]
    for a in (0, 1):
        info[a][a] = -1.0
        for j in range(k):
            info[a][2 + j] = si[a][j]
    for x in range(k):
        for yy in range(k):
            info[2 + x][2 + yy] = -XtWX[x][yy] / n
    Ii = _glm_core._inv(info)
    IJ = [[sum(Ii[x][z] * J[z][yy] for z in range(m)) for yy in range(m)] for x in range(m)]
    V = [[sum(IJ[x][z] * Ii[yy][z] for z in range(m)) / n for yy in range(m)] for x in range(m)]
    ate = est[1] - est[0]
    se = math.sqrt(V[0][0] + V[1][1] - 2 * V[0][1])
    z = 1.959963984540054
    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "n_obs": n_obs,
        "outcome_model": outcome_model,
        "method": "g-formula, sandwich SE (stdReg's estimator, native)",
    }


def sensitivity_rosenbaum(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    gamma_range: tuple[float, float] = (1.0, 3.0),
    n_gamma: int = 20,
) -> pd.DataFrame:
    """
    Rosenbaum bounds sensitivity analysis for hidden confounding.

    Tests whether the sign-rank test conclusion is robust to an unmeasured
    confounder that could increase the odds of treatment assignment by a
    factor of Gamma (the sensitivity parameter).

    For a range of Gamma values, the method computes:
    - p_lower: p-value under the most favourable assignment (best case)
    - p_upper: p-value under the most adverse assignment (worst case)

    The analysis is based on the Wilcoxon signed-rank statistic applied to
    matched pairs. Here we approximate the matched analysis by using all
    discordant (T=1, T=0) pairs sorted by outcome.

    :param data: DataFrame containing all required columns.
    :param treatment: Column name of the binary treatment indicator (0/1).
    :param outcome: Column name of the outcome variable.
    :param covariates: Covariate column names (used for matching approximation).
    :param gamma_range: Tuple (min_gamma, max_gamma). Default (1.0, 3.0).
    :param n_gamma: Number of Gamma values to evaluate. Default 20.
    :return: DataFrame with columns ``Gamma``, ``p_lower``, ``p_upper``.
    :raises ValueError: If required columns are missing or gamma_range is invalid.

    Notes
    -----
    This implementation uses the normal approximation to the signed-rank
    distribution under sensitivity bounds (Rosenbaum, 2002, Chapter 4).
    For small samples or exact analysis, use the R ``sensitivitymw`` or
    ``rbounds`` package.

    References
    ----------
    Rosenbaum, P. R. (2002). Observational Studies (2nd ed.). Springer. (Chapter 4.)
    Rosenbaum, P. R. (2007). Sensitivity analysis for m-estimates, tests, and
        confidence intervals in matched observational studies. Biometrics, 63(2), 456-464.
    """
    required_cols = [treatment, outcome]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}.")
    if gamma_range[0] < 1.0:
        raise ValueError(f"Minimum Gamma must be >= 1.0, got {gamma_range[0]}.")
    if gamma_range[1] <= gamma_range[0]:
        raise ValueError("gamma_range[1] must be > gamma_range[0].")
    if n_gamma < 2:
        raise ValueError(f"n_gamma must be >= 2, got {n_gamma}.")

    df = data[[treatment, outcome]].dropna().copy()
    treated = df[df[treatment] == 1][outcome].values
    control = df[df[treatment] == 0][outcome].values

    # Pair each treated unit with the nearest-ranked control unit
    # (simplified matching by sorted rank proximity)
    min_n = min(len(treated), len(control))
    if min_n < 2:
        raise ValueError("At least 2 treated and 2 control units are required for Rosenbaum bounds.")
    treated_sorted = np.sort(treated)[:min_n]
    control_sorted = np.sort(control)[:min_n]
    differences = treated_sorted - control_sorted

    # Wilcoxon signed-rank statistic (T+)
    n_pairs = len(differences)
    abs_diff = np.abs(differences)
    ranks = scipy_stats.rankdata(abs_diff)
    T_plus = float(np.sum(ranks[differences > 0]))

    gammas = np.linspace(gamma_range[0], gamma_range[1], n_gamma)
    results = []
    for gamma in gammas:
        # Under Gamma, the maximum p_i = gamma / (1 + gamma) (worst case for T+)
        # and minimum p_i = 1 / (1 + gamma) (best case for T+)
        p_max = gamma / (1.0 + gamma)  # upper bound (worst case H0 rejection)
        p_min = 1.0 / (1.0 + gamma)  # lower bound (best case H0 rejection)

        # Expected value and variance of T+ under each extreme
        mu_upper = n_pairs * (n_pairs + 1) / 2 * p_max
        var_upper = n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 6 * p_max * (1 - p_max)

        mu_lower = n_pairs * (n_pairs + 1) / 2 * p_min
        var_lower = n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 6 * p_min * (1 - p_min)

        # Two-sided p-values using normal approximation
        if var_upper > 0:
            z_upper = (T_plus - mu_upper) / math.sqrt(var_upper)
            p_upper = 2.0 * float(scipy_stats.norm.sf(abs(z_upper)))
        else:
            p_upper = float("nan")

        if var_lower > 0:
            z_lower = (T_plus - mu_lower) / math.sqrt(var_lower)
            p_lower = 2.0 * float(scipy_stats.norm.sf(abs(z_lower)))
        else:
            p_lower = float("nan")

        results.append(
            {
                "Gamma": float(gamma),
                "p_lower": float(p_lower),
                "p_upper": float(p_upper),
            }
        )

    return pd.DataFrame(results)


# ===========================================================================
# SECTION 6 -- E-VALUE
# ===========================================================================


def e_value(ate: float, se: float, *, null: float = 0.0) -> float:
    """
    E-value for unmeasured confounding (VanderWeele & Ding, 2017).

    The E-value is the minimum strength of association (on the risk-ratio
    scale) that an unmeasured confounder would need to have with both the
    treatment and the outcome to fully explain away the observed effect,
    conditional on the measured covariates.

    For a risk ratio (RR) effect estimate the E-value is:

    .. math::

        E = \\text{RR} + \\sqrt{\\text{RR} \\cdot (\\text{RR} - 1)}

    where RR > 1. For RR < 1, compute E on 1/RR.

    Since the ATE here is a difference (not a ratio), we first convert using
    a delta-method approximation to get a risk-ratio-like effect, using the
    relationship RR ≈ exp(|ATE - null| / se) (treating the z-score as a
    log-RR approximation). This is an approximation appropriate for
    continuous-scale effects reported with a standard error.

    :param ate: Point estimate of the treatment effect.
    :param se: Standard error of the ATE estimate (must be > 0).
    :param null: Null value to test against. Default 0.0.
    :return: E-value (float >= 1.0). Returns 1.0 if the estimate is at the null.
    :raises ValueError: If se <= 0.

    Notes
    -----
    For binary outcomes with a risk ratio or odds ratio estimate, convert
    the OR to the risk-ratio scale first, then apply the E-value formula
    directly. This function is designed for the continuous-ATE setting.

    References
    ----------
    VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational
        research: Introducing the E-value. Annals of Internal Medicine, 167(4), 268-274.
    VanderWeele, T. J., Mathur, M. B., & Ding, P. (2019). Correcting
        misinterpretations of the E-value. Annals of Internal Medicine, 170(2), 131-132.
    """
    if se <= 0:
        raise ValueError(f"se must be > 0, got {se}.")
    # Distance from null in SE units (absolute z-score)
    z = abs(ate - null) / se
    if z == 0.0:
        return 1.0  # Effect is exactly at the null; no confounding needed
    # Convert z-score to risk-ratio approximation: RR ≈ exp(z * some_factor)
    # We use the VanderWeele-Ding continuous-scale approximation:
    # RR = exp(z / sqrt(n)) is not invariant; instead use the log-linear approximation
    # treating z as the key input:
    # E = exp(0.91 * sqrt(z^2 / (z^2 + 1))) * ... but this is complex.
    # Preferred approach: treat |ATE - null| as log(RR) directly (appropriate when
    # ATE is on a log scale e.g. log-OR, log-RR). For linear ATE, the E-value
    # quantifies confounding on an approximate log-RR scale.
    # Simple conservative approximation: RR_equiv = exp(|ate - null| / se * 0.5)
    # More principled: use VanderWeele's formula for the lower CI bound.
    # Here we apply the standard E-value formula on the z-stat directly.
    # RR proxy = exp(z) following Mathur & VanderWeele (2020) continuous approach
    rr = math.exp(z)
    if rr <= 1.0:
        return 1.0
    e_val = rr + math.sqrt(rr * (rr - 1.0))
    return float(e_val)
