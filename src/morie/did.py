"""
Difference-in-Differences estimators for causal inference.

Implements classic 2x2 DiD, generalized DiD with multiple time periods
(Callaway--Sant'Anna), staggered adoption, event studies, Bacon decomposition,
doubly-robust DiD, triple differences, synthetic DiD, and comprehensive
pre-trend and placebo testing infrastructure.

References
----------
Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-Differences with
multiple time periods. *Journal of Econometrics*, 225(2), 200--230.
https://doi.org/10.1016/j.jeconom.2020.12.001

Goodman-Bacon, A. (2021). Difference-in-differences with variation in
treatment timing. *Journal of Econometrics*, 225(2), 254--277.
https://doi.org/10.1016/j.jeconom.2021.03.014

Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust
difference-in-differences estimators. *Journal of Econometrics*, 219(1),
101--122. https://doi.org/10.1016/j.jeconom.2020.06.003

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
(2021). Synthetic difference-in-differences. *American Economic Review*,
111(12), 4088--4118. https://doi.org/10.1257/aer.20190159
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn import _stats_core as stats


class _MissingDep:
    """Placeholder for a dependency being nativized (task #141)."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        raise ImportError(f"{self._name} is no longer bundled; this code path awaits its native morie implementation")

    def __call__(self, *a, **k):
        raise ImportError(f"{self._name} is no longer bundled; this code path awaits its native morie implementation")


try:
    from morie.fn._ml_core import GradientBoostingClassifier, GradientBoostingRegressor
except ImportError:
    GradientBoostingClassifier = _MissingDep("GradientBoostingClassifier")
    GradientBoostingRegressor = _MissingDep("GradientBoostingRegressor")
try:
    from morie.fn._ml_core import LinearRegression, LogisticRegression
except ImportError:
    LinearRegression = _MissingDep("LinearRegression")
    LogisticRegression = _MissingDep("LogisticRegression")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class DiDResult:
    """Container for a Difference-in-Differences estimate.

    Attributes
    ----------
    estimate : float
        Point estimate of the treatment effect (ATT).
    std_error : float
        Standard error of the estimate.
    t_stat : float
        t-statistic (estimate / std_error).
    p_value : float
        Two-sided p-value.
    ci_lower : float
        Lower bound of the 95 % confidence interval.
    ci_upper : float
        Upper bound of the 95 % confidence interval.
    n_treated : int
        Number of treated observations.
    n_control : int
        Number of control observations.
    method : str
        Name of the estimation method used.
    details : dict
        Additional diagnostic information.
    """

    estimate: float
    std_error: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_treated: int
    n_control: int
    method: str = "did_2x2"
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a one-row summary DataFrame."""
        return pd.DataFrame(
            {
                "estimate": [self.estimate],
                "std_error": [self.std_error],
                "t_stat": [self.t_stat],
                "p_value": [self.p_value],
                "ci_lower": [self.ci_lower],
                "ci_upper": [self.ci_upper],
                "n_treated": [self.n_treated],
                "n_control": [self.n_control],
                "method": [self.method],
            }
        )


@dataclass
class EventStudyResult:
    """Container for event-study coefficient estimates.

    Attributes
    ----------
    coefficients : pd.DataFrame
        DataFrame with columns ``relative_time``, ``estimate``,
        ``std_error``, ``ci_lower``, ``ci_upper``, ``p_value``.
    reference_period : int
        The omitted (normalised-to-zero) relative time period.
    pre_trend_f_stat : float
        Joint F-statistic for pre-treatment coefficients.
    pre_trend_p_value : float
        p-value for the joint pre-trend test.
    """

    coefficients: pd.DataFrame
    reference_period: int
    pre_trend_f_stat: float
    pre_trend_p_value: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaconDecomposition:
    """Container for Goodman-Bacon decomposition.

    Attributes
    ----------
    components : pd.DataFrame
        DataFrame with columns ``group1``, ``group2``, ``estimate``,
        ``weight``, ``type`` (timing, always-treated, never-treated).
    overall_estimate : float
        Weighted-average TWFE DiD estimate.
    """

    components: pd.DataFrame
    overall_estimate: float


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _ols_robust_se(
    X: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray | None = None,
    return_vcov: bool = False,
):
    """OLS with heteroskedasticity- or cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n, k) **including** intercept column if desired.
    y : np.ndarray
        Outcome vector (n,).
    cluster_ids : np.ndarray, optional
        Cluster identifiers for cluster-robust (CR1) variance.

    Returns
    -------
    beta : np.ndarray
        Coefficient vector (k,).
    se : np.ndarray
        Robust standard errors (k,).
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    if cluster_ids is not None:
        unique_clusters = np.unique(cluster_ids)
        g = len(unique_clusters)
        meat = np.zeros((k, k))
        for c in unique_clusters:
            mask = cluster_ids == c
            Xc = X[mask]
            ec = resid[mask]
            score = Xc.T @ ec  # (k,)
            meat += np.outer(score, score)
        # Small-sample correction (CR1)
        correction = (g / (g - 1)) * ((n - 1) / (n - k))
        V = correction * XtX_inv @ meat @ XtX_inv
    else:
        # HC1 robust variance
        meat = X.T @ np.diag(resid**2) @ X
        correction = n / (n - k)
        V = correction * XtX_inv @ meat @ XtX_inv

    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    if return_vcov:
        return beta, se, V
    return beta, se


def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend an intercept column of ones."""
    return np.column_stack([np.ones(X.shape[0]), X])


def _twfe_within(df, unit, time, columns):
    """Within transformation for unit and time fixed effects by alternating
    projections, exact for unbalanced panels (one pass of two-way demeaning
    is only right for balanced ones). Returns one array per column."""
    uid = df[unit].tolist()
    tid = df[time].tolist()
    ug, tg = {}, {}
    for i in range(len(uid)):
        ug.setdefault(uid[i], []).append(i)
        tg.setdefault(tid[i], []).append(i)
    out = []
    for col in columns:
        v = [float(a) for a in col]
        for _ in range(100000):
            delta = 0.0
            for groups in (ug, tg):
                for idx in groups.values():
                    m = sum(v[i] for i in idx) / len(idx)
                    if m != 0.0:
                        delta = max(delta, abs(m))
                        for i in idx:
                            v[i] -= m
            if delta < 1e-13:
                break
        out.append(np.array(v))
    return out


def _twfe_cluster_vcov(X, resid, cl, uid, tid):
    """Cluster-robust (CR1) variance of a unit and time FE regression with
    fixest's default small-sample factor, ssc(adj = TRUE, fixef.K =
    "nested"): G/(G-1) (n-1)/(n-K), K the regressors plus the fixed-effect
    levels less one per extra FE, a FE nested in the cluster counting once."""
    n, k = X.shape

    def nested(ids):
        owner = {}
        return all(owner.setdefault(a_, c_) == c_ for a_, c_ in zip(ids, cl))

    K = k + sum(1 if nested(ids) else len(set(ids)) for ids in (uid, tid)) - 1
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    groups = {}
    for i, c_ in enumerate(cl):
        groups.setdefault(c_, []).append(i)
    for idx in groups.values():
        sc = X[idx].T @ resid[idx]
        meat += np.outer(sc, sc)
    G = len(groups)
    return (G / (G - 1)) * ((n - 1) / (n - K)) * XtX_inv @ meat @ XtX_inv, G


def _make_ci(
    estimate: float,
    se: float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Construct a symmetric normal-approximation CI."""
    z = stats.norm.ppf(1 - alpha / 2)
    return estimate - z * se, estimate + z * se


# ---------------------------------------------------------------------------
# 1. Classic 2x2 DiD
# ---------------------------------------------------------------------------


def did_2x2(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    post: str,
    *,
    covariates: list[str] | None = None,
    cluster: str | None = None,
    alpha: float = 0.05,
) -> DiDResult:
    r"""Estimate a classic 2x2 Difference-in-Differences treatment effect.

    The canonical DiD estimator for settings with two groups (treated vs
    control) and two periods (pre vs post).  With no covariates the
    estimator is:

    .. math::

        \hat\tau_{\text{DiD}} =
        \bigl(\bar Y_{1,\text{post}} - \bar Y_{1,\text{pre}}\bigr)
        - \bigl(\bar Y_{0,\text{post}} - \bar Y_{0,\text{pre}}\bigr)

    When *covariates* are provided the regression specification is:

    .. math::

        Y_{it} = \alpha + \beta\,D_i + \gamma\,\text{Post}_t
        + \tau\,(D_i \times \text{Post}_t) + X_{it}'\delta + \varepsilon_{it}

    and :math:`\hat\tau` is the coefficient on the interaction term.

    Parameters
    ----------
    data : pd.DataFrame
        Panel or repeated cross-section data.
    outcome : str
        Name of the outcome column.
    treatment : str
        Binary (0/1) column indicating the treatment group.
    post : str
        Binary (0/1) column indicating the post-treatment period.
    covariates : list of str, optional
        Additional covariates for the regression specification.
    cluster : str, optional
        Column for cluster-robust standard errors.
    alpha : float
        Significance level for confidence intervals (default 0.05).

    Returns
    -------
    DiDResult
        Estimation results including point estimate, SE, CI, and diagnostics.

    References
    ----------
    Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless
    Econometrics*. Princeton University Press.
    """
    df = data.dropna(subset=[outcome, treatment, post]).copy()
    d = df[treatment].values.astype(float)
    p = df[post].values.astype(float)
    y = df[outcome].values.astype(float)
    interaction = d * p

    if covariates:
        X_cov = df[covariates].values.astype(float)
        X = _add_intercept(np.column_stack([d, p, interaction, X_cov]))
    else:
        X = _add_intercept(np.column_stack([d, p, interaction]))

    cluster_ids = df[cluster].values if cluster else None
    beta, se = _ols_robust_se(X, y, cluster_ids=cluster_ids)

    # The interaction coefficient is at index 3 (intercept=0, d=1, p=2, interaction=3)
    tau_idx = 3
    est = float(beta[tau_idx])
    se_est = float(se[tau_idx])
    t_val = est / se_est if se_est > 0 else 0.0
    p_val = float(2 * stats.norm.sf(abs(t_val)))
    ci_lo, ci_hi = _make_ci(est, se_est, alpha)

    return DiDResult(
        estimate=est,
        std_error=se_est,
        t_stat=t_val,
        p_value=p_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_treated=int(d.sum()),
        n_control=int((1 - d).sum()),
        method="did_2x2",
        details={
            "all_coefficients": beta.tolist(),
            "all_se": se.tolist(),
            "n_obs": len(y),
        },
    )


# ---------------------------------------------------------------------------
# 2. Repeated cross-section DiD
# ---------------------------------------------------------------------------


def did_repeated_cross_section(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    post: str,
    *,
    covariates: list[str] | None = None,
    weights: str | None = None,
    cluster: str | None = None,
    alpha: float = 0.05,
) -> DiDResult:
    """DiD estimator designed for repeated cross-section data.

    Identical to :func:`did_2x2` but optionally incorporates survey
    weights.  When *weights* is provided, weighted least squares is used
    for estimation.

    Parameters
    ----------
    data : pd.DataFrame
        Repeated cross-section data.
    outcome, treatment, post : str
        Column names for outcome, treatment indicator, and post indicator.
    covariates : list of str, optional
        Additional covariates.
    weights : str, optional
        Column containing survey / sampling weights.
    cluster : str, optional
        Column for cluster-robust SE.
    alpha : float
        Significance level.

    Returns
    -------
    DiDResult
    """
    df = data.dropna(subset=[outcome, treatment, post]).copy()
    d = df[treatment].values.astype(float)
    p = df[post].values.astype(float)
    y = df[outcome].values.astype(float)
    interaction = d * p

    if covariates:
        X_cov = df[covariates].values.astype(float)
        X = _add_intercept(np.column_stack([d, p, interaction, X_cov]))
    else:
        X = _add_intercept(np.column_stack([d, p, interaction]))

    if weights is not None:
        w = np.sqrt(df[weights].values.astype(float))
        X = X * w[:, None]
        y = y * w

    cluster_ids = df[cluster].values if cluster else None
    beta, se = _ols_robust_se(X, y, cluster_ids=cluster_ids)

    tau_idx = 3
    est = float(beta[tau_idx])
    se_est = float(se[tau_idx])
    t_val = est / se_est if se_est > 0 else 0.0
    p_val = float(2 * stats.norm.sf(abs(t_val)))
    ci_lo, ci_hi = _make_ci(est, se_est, alpha)

    return DiDResult(
        estimate=est,
        std_error=se_est,
        t_stat=t_val,
        p_value=p_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_treated=int((df[treatment] == 1).sum()),
        n_control=int((df[treatment] == 0).sum()),
        method="did_repeated_cross_section",
        details={"all_coefficients": beta.tolist(), "n_obs": len(df)},
    )


# ---------------------------------------------------------------------------
# 3. Panel fixed-effects DiD
# ---------------------------------------------------------------------------


def did_panel_fe(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    unit: str,
    time: str,
    *,
    covariates: list[str] | None = None,
    cluster: str | None = None,
    alpha: float = 0.05,
) -> DiDResult:
    """DiD with unit and time fixed effects (two-way fixed effects, TWFE).

    Demeans the outcome by unit and time means (within transformation)
    and regresses on the treatment indicator.

    Parameters
    ----------
    data : pd.DataFrame
        Balanced or unbalanced panel data.
    outcome : str
        Outcome column.
    treatment : str
        Binary column (0/1) indicating treatment status in each period.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    covariates : list of str, optional
        Time-varying covariates.
    cluster : str, optional
        Cluster variable for standard errors (defaults to *unit*).
    alpha : float
        Significance level.

    Returns
    -------
    DiDResult
    """
    df = data.dropna(subset=[outcome, treatment, unit, time]).copy()
    cols_raw = [df[outcome].values.astype(float).tolist(), df[treatment].values.astype(float).tolist()]
    if covariates:
        cols_raw += [df[c].values.astype(float).tolist() for c in covariates]
    dm = _twfe_within(df, unit, time, cols_raw)
    y_demean = dm[0]
    X = np.column_stack(dm[1:])
    cl = df[cluster if cluster else unit].tolist()
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y_demean)
    V, _ = _twfe_cluster_vcov(X, y_demean - X @ beta, cl, df[unit].tolist(), df[time].tolist())
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    est = float(beta[0])
    se_est = float(se[0])
    t_val = est / se_est if se_est > 0 else 0.0
    p_val = float(2 * stats.norm.sf(abs(t_val)))
    ci_lo, ci_hi = _make_ci(est, se_est, alpha)

    n_treat = int(df[treatment].sum())
    return DiDResult(
        estimate=est,
        std_error=se_est,
        t_stat=t_val,
        p_value=p_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_treated=n_treat,
        n_control=len(df) - n_treat,
        method="did_panel_fe",
        details={"n_units": df[unit].nunique(), "n_periods": df[time].nunique()},
    )


# ---------------------------------------------------------------------------
# 4. Event study specification
# ---------------------------------------------------------------------------


def event_study(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treatment_time: str,
    *,
    covariates: list[str] | None = None,
    reference_period: int = -1,
    leads: int = 4,
    lags: int = 4,
    cluster: str | None = None,
    alpha: float = 0.05,
) -> EventStudyResult:
    r"""Estimate an event-study specification around treatment onset.

    Constructs relative-time dummies :math:`\{1[t - g = k]\}` for
    :math:`k \in [-\text{leads}, \text{lags}]` and regresses the
    outcome on these indicators with unit and time fixed effects.  The
    *reference_period* dummy is omitted (normalised to zero).

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with a treatment-onset column.
    outcome : str
        Outcome column.
    unit : str
        Unit identifier column.
    time : str
        Calendar-time column (integer-valued).
    treatment_time : str
        Column giving the period in which each unit first received
        treatment (``np.inf`` or ``NaN`` for never-treated units).
    covariates : list of str, optional
        Time-varying covariates.
    reference_period : int
        Relative-time period omitted as baseline (default -1).
    leads : int
        Number of pre-treatment periods to include.
    lags : int
        Number of post-treatment periods to include.
    cluster : str, optional
        Cluster variable (defaults to *unit*).
    alpha : float
        Significance level.

    Returns
    -------
    EventStudyResult
    """
    df = data.copy()
    df["_rel_time"] = df[time].astype(float) - df[treatment_time].astype(float)
    # relative-time dummies; the endpoints are binned, so every treated
    # observation outside the window counts in -leads or lags rather than
    # silently in the reference period (Schmidheiny & Siegloch 2020)
    periods = [k for k in range(-leads, lags + 1) if k != reference_period]
    rel = [min(max(v, -leads), lags) if v == v else v for v in df["_rel_time"].tolist()]
    for k in periods:
        df[f"_rel_{k}"] = [1.0 if r_ == k else 0.0 for r_ in rel]
    X_cols = [f"_rel_{k}" for k in periods]
    if covariates:
        X_cols.extend(covariates)
    dm = _twfe_within(
        df,
        unit,
        time,
        [df[outcome].values.astype(float).tolist()] + [df[c].values.astype(float).tolist() for c in X_cols],
    )
    y_dm = dm[0]
    X_dm = np.column_stack(dm[1:])
    cl = df[cluster if cluster else unit].tolist()
    beta = np.linalg.pinv(X_dm.T @ X_dm) @ (X_dm.T @ y_dm)
    V, G = _twfe_cluster_vcov(X_dm, y_dm - X_dm @ beta, cl, df[unit].tolist(), df[time].tolist())
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    coefs = []
    for i, k in enumerate(periods):
        est_k = float(beta[i])
        se_k = float(se[i])
        ci_lo, ci_hi = _make_ci(est_k, se_k, alpha)
        p_k = float(2 * stats.norm.sf(abs(est_k / se_k))) if se_k > 0 else 1.0
        coefs.append(
            {
                "relative_time": k,
                "estimate": est_k,
                "std_error": se_k,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "p_value": p_k,
            }
        )

    # Add the reference period (zero by construction)
    coefs.append(
        {
            "relative_time": reference_period,
            "estimate": 0.0,
            "std_error": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value": np.nan,
        }
    )
    coef_df = pd.DataFrame(coefs).sort_values("relative_time").reset_index(drop=True)

    # Joint pre-trend test: F-test that all pre-treatment coefficients are zero
    pre_indices = [i for i, k in enumerate(periods) if k < 0]
    if len(pre_indices) > 0:
        # joint Wald test with the full cluster covariance, as fixest::wald:
        # F = b' V^-1 b / q on (q, G - 1) degrees of freedom
        pre_beta = beta[pre_indices]
        Vpre = V[np.ix_(pre_indices, pre_indices)]
        q_ = len(pre_indices)
        f_stat = float(pre_beta @ np.linalg.solve(Vpre, pre_beta)) / q_
        f_p = float(stats.f.sf(f_stat, q_, G - 1))
    else:
        f_stat = np.nan
        f_p = np.nan

    return EventStudyResult(
        coefficients=coef_df,
        reference_period=reference_period,
        pre_trend_f_stat=f_stat,
        pre_trend_p_value=f_p,
    )


# ---------------------------------------------------------------------------
# 5. Pre-trend testing
# ---------------------------------------------------------------------------


def test_parallel_trends(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    time: str,
    *,
    unit: str | None = None,
    cluster: str | None = None,
    pre_periods: list | None = None,
) -> dict[str, Any]:
    """Test the parallel trends assumption using pre-treatment data.

    Runs two tests:

    1. **Individual coefficient test**: regresses the outcome on
       group-by-time interactions in the pre-period and reports each
       coefficient.
    2. **Joint F-test**: tests that all interaction coefficients are
       jointly zero.

    Parameters
    ----------
    data : pd.DataFrame
        Panel or repeated cross-section data.
    outcome : str
        Outcome column name.
    treatment : str
        Treatment group indicator.
    time : str
        Time column (integer-valued).
    unit : str, optional
        Unit identifier for panel demeaning.
    cluster : str, optional
        Cluster variable for robust SE.
    pre_periods : list, optional
        Explicit list of pre-treatment time values to test.

    Returns
    -------
    dict
        ``coefficients`` (pd.DataFrame), ``joint_f_stat``, ``joint_p_value``,
        ``parallel_trends_plausible`` (bool, True when p > 0.05).
    """
    df = data.copy()
    all_times = sorted(df[time].unique())

    if pre_periods is None:
        # Assume periods before the first treated observation are pre-treatment
        treated_times = df.loc[df[treatment] == 1, time]
        if len(treated_times) == 0:
            raise ValueError("No treated observations found.")
        first_treat = treated_times.min()
        pre_periods = [t for t in all_times if t < first_treat]

    if len(pre_periods) < 2:
        return {
            "coefficients": pd.DataFrame(),
            "joint_f_stat": np.nan,
            "joint_p_value": np.nan,
            "parallel_trends_plausible": True,
        }

    df_pre = df[df[time].isin(pre_periods)].copy()
    test_periods = pre_periods[1:]

    # Construct interaction dummies
    d_vals = df_pre[treatment].values.astype(float)
    y_vals = df_pre[outcome].values.astype(float)
    interact_cols = []
    for tp in test_periods:
        col = d_vals * (df_pre[time].values == tp).astype(float)
        interact_cols.append(col)

    # Also include main effects
    time_dummies = []
    for tp in test_periods:
        time_dummies.append((df_pre[time].values == tp).astype(float))

    X = _add_intercept(np.column_stack([d_vals] + time_dummies + interact_cols))

    # cluster by the named column, else by the panel unit when given
    cl_col = cluster or unit
    cluster_ids = df_pre[cl_col].values if cl_col else None
    beta, se, V = _ols_robust_se(X, y_vals, cluster_ids=cluster_ids, return_vcov=True)

    # Interaction coefficients start after: intercept (1) + treat (1) + time dummies
    start_idx = 1 + 1 + len(test_periods)
    coefs = []
    for i, tp in enumerate(test_periods):
        idx = start_idx + i
        est_k = float(beta[idx])
        se_k = float(se[idx])
        t_k = est_k / se_k if se_k > 0 else 0.0
        p_k = float(2 * stats.norm.sf(abs(t_k)))
        coefs.append({"period": tp, "estimate": est_k, "std_error": se_k, "t_stat": t_k, "p_value": p_k})

    coef_df = pd.DataFrame(coefs)

    # Joint test
    # joint Wald test with the full robust covariance, as fixest::wald:
    # F = b' V^-1 b / q on (q, G - 1) df when clustered, (q, n - k) otherwise
    idx = list(range(start_idx, start_idx + len(test_periods)))
    ib = beta[idx]
    q_ = len(idx)
    f_joint = float(ib @ np.linalg.solve(V[np.ix_(idx, idx)], ib)) / q_
    df2 = (len(set(cluster_ids.tolist())) - 1) if cluster_ids is not None else (X.shape[0] - X.shape[1])
    joint_p = float(stats.f.sf(f_joint, q_, df2))

    return {
        "coefficients": coef_df,
        "joint_f_stat": f_joint,
        "joint_p_value": joint_p,
        "parallel_trends_plausible": joint_p > 0.05,
    }


# ---------------------------------------------------------------------------
# 6. Parallel trends visualisation data
# ---------------------------------------------------------------------------


def parallel_trends_data(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    time: str,
    *,
    weights: str | None = None,
) -> pd.DataFrame:
    """Compute group-level outcome means over time for parallel-trends plots.

    Parameters
    ----------
    data : pd.DataFrame
        Panel or repeated cross-section.
    outcome, treatment, time : str
        Column names.
    weights : str, optional
        Survey weight column.

    Returns
    -------
    pd.DataFrame
        Columns: ``time``, ``group`` (0 or 1), ``mean_outcome``, ``se``, ``n``.
    """
    df = data.dropna(subset=[outcome, treatment, time]).copy()
    records = []

    for (t_val, g_val), grp in df.groupby([time, treatment]):
        y = grp[outcome].values.astype(float)
        if weights and weights in grp.columns:
            w = grp[weights].values.astype(float)
            w = w / w.sum()
            mean_y = float(np.average(y, weights=w))
            se_y = float(np.sqrt(np.average((y - mean_y) ** 2, weights=w) / len(y)))
        else:
            mean_y = float(y.mean())
            se_y = float(y.std(ddof=1) / np.sqrt(len(y))) if len(y) > 1 else 0.0
        records.append({"time": t_val, "group": g_val, "mean_outcome": mean_y, "se": se_y, "n": len(y)})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 7. Generalized DiD / Callaway-Sant'Anna (group-time ATTs)

# ---------------------------------------------------------------------------
# Sant'Anna-Zhao (2020) panel engines and Callaway-Sant'Anna aggregation
# (mirrors R/did_native.R; equal to DRDID / did::att_gt / did::aggte)
# ---------------------------------------------------------------------------


def _did_ps_fit(D, X):
    """Logistic propensity MLE and its influence-function linear rep."""
    from morie.fn.ps_fit import _ps_irls_beta

    beta = np.array(_ps_irls_beta(X.tolist(), D.tolist()))
    ps = 1.0 / (1.0 + np.exp(-(X @ beta)))
    ps = np.minimum(ps, 1 - 1e-16)
    n = len(D)
    score = (D - ps)[:, None] * X
    hess = (X * (ps * (1 - ps))[:, None]).T @ X / n
    return ps, score @ np.linalg.pinv(hess)


def _did_or_fit(y, X, w):
    """Weighted least squares of y on X; fitted values and linear rep."""
    n = len(y)
    wX = X * w[:, None]
    XpX_inv = np.linalg.pinv(wX.T @ X / n)
    beta = XpX_inv @ (wX.T @ y / n)
    fitted = X @ beta
    return fitted, ((w * (y - fitted))[:, None] * X) @ XpX_inv


def _did_panel_att(dy, D, X, est_method):
    """ATT and influence function of DRDID's panel dr / reg / ipw estimators."""
    w_treat = D
    if est_method == "reg":
        m, lin_or = _did_or_fit(dy, X, 1.0 - D)
        att_t = np.mean(w_treat * dy) / np.mean(w_treat)
        att_c = np.mean(w_treat * m) / np.mean(w_treat)
        inf_t = (w_treat * dy - w_treat * att_t) / np.mean(w_treat)
        M1 = np.mean(w_treat[:, None] * X, axis=0) / np.mean(w_treat)
        inf_c = (w_treat * m - w_treat * att_c) / np.mean(w_treat) + lin_or @ M1
        return float(att_t - att_c), inf_t - inf_c
    ps, lin_ps = _did_ps_fit(D, X)
    w_cont = ps * (1 - D) / (1 - ps)
    if est_method == "ipw":
        att_t = np.mean(w_treat * dy) / np.mean(w_treat)
        att_c = np.mean(w_cont * dy) / np.mean(w_cont)
        inf_t = (w_treat * dy - w_treat * att_t) / np.mean(w_treat)
        inf_c1 = (w_cont * dy - w_cont * att_c) / np.mean(w_cont)
        M2 = np.mean((w_cont * (dy - att_c))[:, None] * X, axis=0) / np.mean(w_cont)
        return float(att_t - att_c), inf_t - inf_c1 - lin_ps @ M2
    m, lin_or = _did_or_fit(dy, X, 1.0 - D)
    r = dy - m
    att_t = np.mean(w_treat * r) / np.mean(w_treat)
    att_c = np.mean(w_cont * r) / np.mean(w_cont)
    M1 = np.mean(w_treat[:, None] * X, axis=0) / np.mean(w_treat)
    inf_t = (w_treat * r - w_treat * att_t) / np.mean(w_treat) - lin_or @ M1
    M2 = np.mean((w_cont * (r - att_c))[:, None] * X, axis=0) / np.mean(w_cont)
    M3 = np.mean(w_cont[:, None] * X, axis=0) / np.mean(w_cont)
    inf_c = (w_cont * r - w_cont * att_c) / np.mean(w_cont) + lin_ps @ M2 - lin_or @ M3
    return float(att_t - att_c), inf_t - inf_c


def _did_mboot_se(IF, biters, seed):
    """Mammen multiplier-bootstrap SE (did::mboot): IQR / (z.75 - z.25)."""
    IF = np.asarray(IF, dtype=float)
    if IF.ndim == 1:
        IF = IF[:, None]
    n = IF.shape[0]
    rng = np.random.default_rng(seed)
    sq5 = math.sqrt(5)
    k1, k2, p1 = 0.5 * (1 - sq5), 0.5 * (1 + sq5), 0.5 * (1 + sq5) / sq5
    boot = np.array(
        [math.sqrt(n) * np.mean(np.where(rng.random(n) < p1, k1, k2)[:, None] * IF, axis=0) for _ in range(int(biters))]
    )
    q = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)
    return (np.quantile(boot, 0.75, axis=0) - np.quantile(boot, 0.25, axis=0)) / q / math.sqrt(n)


def _did_aggte(fit, kind):
    """did::aggte on a group_time_att fit: simple / group / dynamic / calendar."""
    grp, tt, att = fit["group"], fit["t"], fit["att"]
    IF, gu, n = fit["IF"], fit["G"], fit["n"]
    glist = np.array(sorted(set(grp.tolist())))
    pgg = np.array([np.mean(gu == g) for g in glist])
    pg = pgg[np.searchsorted(glist, grp)]

    def se_of(inf):
        if fit["biters"] > 0:
            se = float(_did_mboot_se(inf, fit["biters"], fit["seed"])[0])
        else:
            se = math.sqrt(float(np.mean(inf**2)) / n)
        return se if np.isfinite(se) and se > math.sqrt(2.220446049250313e-16) * 10 else float("nan")

    def wif(keep, pgv, gv):
        s_ = float(np.sum(pgv[keep]))
        c = np.column_stack([(gu == gv[k]).astype(float) - pgv[k] for k in keep])
        return c / s_ - np.outer(np.sum(c, axis=1), pgv[keep] / s_**2)

    def agg_if(a, inf, keep, w, wf=None):
        out = inf[:, keep] @ w
        return out + wf @ a[keep] if wf is not None else out

    def one(keep, with_wif=True):
        keep = np.asarray(keep, dtype=int)
        w = pg[keep] / np.sum(pg[keep])
        inf = agg_if(att, IF, keep, w, wif(keep, pg, grp) if with_wif else None)
        return float(np.sum(w * att[keep])), inf, se_of(inf)

    if kind == "simple":
        est, _, se = one(np.where(grp <= tt)[0])
        return est, se, None, None, None
    if kind == "group":
        parts = [one(np.where((grp == g) & (g <= tt))[0], with_wif=False) for g in glist]
        est_g = np.array([p_[0] for p_ in parts])
        inf_g = np.column_stack([p_[1] for p_ in parts])
        k = np.arange(len(glist))
        inf = agg_if(est_g, inf_g, k, pgg / np.sum(pgg), wif(k, pgg, glist))
        return float(np.sum(est_g * pgg) / np.sum(pgg)), se_of(inf), glist, est_g, np.array([p_[2] for p_ in parts])
    if kind == "dynamic":
        e = tt - grp
        keys = np.array(sorted(set(e.tolist())))
        parts = [one(np.where(e == ee)[0]) for ee in keys]
    elif kind == "calendar":
        keys = np.array([t1 for t1 in sorted(set(tt[tt >= np.min(grp)].tolist())) if np.any((tt == t1) & (grp <= tt))])
        parts = [one(np.where((tt == t1) & (grp <= tt))[0]) for t1 in keys]
    else:
        raise ValueError(f"Unknown aggregation: {kind}")
    est_k = np.array([p_[0] for p_ in parts])
    inf_k = np.column_stack([p_[1] for p_ in parts])
    k = np.where(keys >= 0)[0] if kind == "dynamic" else np.arange(len(keys))
    inf = agg_if(est_k, inf_k, k, np.full(len(k), 1.0 / len(k)))
    return float(np.mean(est_k[k])), se_of(inf), keys, est_k, np.array([p_[2] for p_ in parts])


# ---------------------------------------------------------------------------


def _outcome_regression_att(
    y: np.ndarray,
    X: np.ndarray,
    treat: np.ndarray,
) -> float:
    """Outcome-regression ATT: E[Y(1)-Y(0)|D=1] via linear imputation."""
    model = LinearRegression().fit(X[treat == 0], y[treat == 0])
    y0_hat = model.predict(X[treat == 1])
    return float(np.mean(y[treat == 1] - y0_hat))


def _odds_weights(treat: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """Control odds weights p/(1-p), zero for treated units and for
    controls with p >= 0.995 (DRDID's trimming rule, Sant'Anna & Zhao
    2020); no other clipping."""
    ctrl = (treat == 0) & (ps < 0.995)
    w = np.zeros(len(ps))
    w[ctrl] = ps[ctrl] / (1.0 - ps[ctrl])
    return w


def _ipw_att(
    y: np.ndarray,
    treat: np.ndarray,
    ps: np.ndarray,
) -> float:
    """Normalised (Hajek) IPW ATT, as DRDID::std_ipw_did_panel."""
    n1 = treat.sum()
    if n1 == 0:
        return 0.0
    w = _odds_weights(treat, ps)
    return float(np.mean(y[treat == 1]) - np.sum(w * y) / np.sum(w))


def group_time_att(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treatment_time: str,
    *,
    covariates: list[str] | None = None,
    method: str = "doubly_robust",
    control_group: str = "never_treated",
    n_bootstrap: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
    se_convention: str = "reference",
) -> pd.DataFrame:
    r"""Estimate group-time average treatment effects (Callaway & Sant'Anna 2021).

    For each cohort :math:`g` (units first treated at time :math:`g`) and
    each post-treatment calendar period :math:`t`, estimates
    :math:`\text{ATT}(g, t)`.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Outcome column.
    unit : str
        Unit identifier.
    time : str
        Calendar-time column (integer).
    treatment_time : str
        Column with treatment-onset period (use ``np.inf`` for
        never-treated).
    covariates : list of str, optional
        Covariates for doubly-robust estimation.
    method : str
        ``"doubly_robust"`` (default), ``"ipw"``, or ``"outcome_regression"``.
    control_group : str
        ``"never_treated"`` or ``"not_yet_treated"``.
    n_bootstrap : int
        Mammen multiplier-bootstrap replications (0 = analytic
        influence-function standard errors).
    seed : int
        Random seed for the multiplier bootstrap.
    alpha : float
        Significance level.
    se_convention : str
        Analytic SE: ``"reference"`` (did's sqrt(mean(IF^2)/n)) or
        ``"bessel"`` (sd(IF)/sqrt(n)).

    Returns
    -------
    pd.DataFrame
        Columns: ``cohort``, ``time``, ``att``, ``std_error``,
        ``ci_lower``, ``ci_upper``, ``p_value``, ``n_treated``, ``post``;
        ``attrs["did_fit"]`` holds the influence functions used by
        :func:`aggregate_gt_att`.

    Notes
    -----
    Base period is ``g - 1`` after treatment and ``t - 1`` before it;
    each cell is DRDID's panel estimator (dr / reg / ipw) with the
    covariates of the base period. Point estimates and analytic SEs equal
    ``did::att_gt(..., bstrap = FALSE)``.

    References
    ----------
    Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-Differences
    with multiple time periods. *Journal of Econometrics*, 225(2), 200--230.
    """
    est_method = {"doubly_robust": "dr", "ipw": "ipw", "outcome_regression": "reg"}.get(method, "dr")
    df = pd._coerce_frame(data)
    g_all = np.array(df[treatment_time].astype(float).values)
    g_all = np.where(np.isfinite(g_all), g_all, 0.0)  # never treated = 0
    time_all = np.array(df[time].values, dtype=float)
    y_all = np.array(df[outcome].values, dtype=float)
    unit_all = list(df[unit].values)
    ids = sorted(set(unit_all))
    pos = {u: i for i, u in enumerate(ids)}
    uid_all = np.array([pos[u] for u in unit_all])
    n_ids = len(ids)
    Xcov_all = np.array(df[list(covariates)].values, dtype=float) if covariates else None
    tlist = sorted(set(time_all.tolist()))
    glist = [g for g in sorted(set(g_all[g_all > 0].tolist())) if g > tlist[0]]
    rows, IF_cols = [], []
    for g in glist:
        for tt in tlist[1:]:
            # base period: g - 1 after treatment, varying t - 1 before
            pret = max(t_ for t_ in tlist if t_ < (g if tt >= g else tt))
            if control_group == "never_treated":
                is_control = g_all == 0
            else:
                is_control = (g_all == 0) | ((g_all > max(tt, pret)) & (g_all != g))
            cidx = np.where(((g_all == g) | is_control) & ((time_all == pret) | (time_all == tt)))[0]
            counts = np.bincount(uid_all[cidx], minlength=n_ids)
            cidx = cidx[counts[uid_all[cidx]] == 2]
            if len(cidx) == 0:
                continue
            cidx = cidx[np.lexsort((time_all[cidx], uid_all[cidx]))]
            pre_idx = cidx[time_all[cidx] == pret]
            dy = y_all[cidx[time_all[cidx] == tt]] - y_all[pre_idx]
            D = (g_all[pre_idx] == g).astype(float)
            if D.sum() == 0 or (1 - D).sum() == 0:
                continue
            X = np.ones((len(dy), 1))
            if Xcov_all is not None:
                X = np.column_stack([X, Xcov_all[pre_idx]])
            att_gt, inf = _did_panel_att(dy, D, X, est_method)
            # map to the full unit list, scaled by n / n_sub (did's convention)
            IF_full = np.zeros(n_ids)
            IF_full[uid_all[pre_idx]] = inf * (n_ids / len(dy))
            if se_convention == "bessel":
                se_a = float(np.std(IF_full, ddof=1)) / math.sqrt(n_ids)
            else:
                se_a = math.sqrt(float(np.mean(IF_full**2)) / n_ids)
            rows.append((g, tt, att_gt, int(D.sum()), se_a))
            IF_cols.append(IF_full)
    cols = ["cohort", "time", "att", "std_error", "ci_lower", "ci_upper", "p_value", "n_treated", "post"]
    if not rows:
        return pd.DataFrame({c: [] for c in cols})
    IF = np.column_stack(IF_cols)
    se = np.array([r[4] for r in rows])
    if n_bootstrap > 0:
        sb = _did_mboot_se(IF, n_bootstrap, seed)
        se = np.where(np.isfinite(sb) & (sb > 0), sb, se)
    z = stats.norm.ppf(1 - alpha / 2)
    grp = np.array([r[0] for r in rows])
    tcol = np.array([r[1] for r in rows])
    att = np.array([r[2] for r in rows])
    out = pd.DataFrame(
        {
            "cohort": grp.tolist(),
            "time": tcol.tolist(),
            "att": att.tolist(),
            "std_error": se.tolist(),
            "ci_lower": (att - z * se).tolist(),
            "ci_upper": (att + z * se).tolist(),
            "p_value": (2 * stats.norm.sf(np.abs(att / se))).tolist(),
            "n_treated": [r[3] for r in rows],
            "post": (tcol >= grp).tolist(),
        }
    )
    G_unit = np.zeros(n_ids)
    G_unit[uid_all] = g_all
    out.attrs["did_fit"] = {
        "group": grp,
        "t": tcol,
        "att": att,
        "IF": IF,
        "G": G_unit,
        "n": n_ids,
        "biters": int(n_bootstrap),
        "seed": seed,
        "alpha": alpha,
    }
    return out


# ---------------------------------------------------------------------------
# 8. Aggregate group-time ATTs
# ---------------------------------------------------------------------------


def aggregate_gt_att(
    gt_results: pd.DataFrame,
    *,
    aggregation: str = "overall",
    time_col: str = "time",
    cohort_col: str = "cohort",
    att_col: str = "att",
    se_col: str = "std_error",
) -> pd.DataFrame:
    """Aggregate group-time ATTs into summary treatment effect parameters.

    On the unmodified output of :func:`group_time_att` the aggregation
    uses its influence functions and equals ``did::aggte`` (simple,
    group, dynamic, calendar), estimated-weight term included; any other
    table is combined as independent cell estimates.

    Parameters
    ----------
    gt_results : pd.DataFrame
        Output of :func:`group_time_att`.
    aggregation : str
        ``"overall"`` (single ATT), ``"cohort"`` (by cohort),
        ``"calendar_time"`` (by period), or ``"event_time"``
        (by relative period since treatment).
    time_col, cohort_col, att_col, se_col : str
        Column name overrides.

    Returns
    -------
    pd.DataFrame
        Aggregated estimates with ``estimate``, ``std_error``, ``ci_lower``,
        ``ci_upper``.
    """
    fit = getattr(gt_results, "attrs", {}).get("did_fit")
    if fit is not None and len(gt_results) == len(fit["att"]):
        # influence-function aggregation, as did::aggte
        kind = {"overall": "simple", "cohort": "group", "calendar_time": "calendar", "event_time": "dynamic"}.get(
            aggregation
        )
        if kind is None:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        est, se, keys, est_k, se_k = _did_aggte(fit, kind)
        z = stats.norm.ppf(1 - fit["alpha"] / 2)
        if kind == "simple":
            return pd.DataFrame(
                [
                    {
                        "group": "overall",
                        "estimate": est,
                        "std_error": se,
                        "ci_lower": est - z * se,
                        "ci_upper": est + z * se,
                    }
                ]
            )
        return pd.DataFrame(
            {
                "group": keys.tolist(),
                "estimate": est_k.tolist(),
                "std_error": se_k.tolist(),
                "ci_lower": (est_k - z * se_k).tolist(),
                "ci_upper": (est_k + z * se_k).tolist(),
            }
        )
    df = gt_results.copy()
    df["_rel_time"] = df[time_col] - df[cohort_col]

    if aggregation == "overall":
        est = df[att_col].mean()
        se = float(np.sqrt(np.mean(df[se_col] ** 2) / len(df)))
        ci_lo, ci_hi = _make_ci(est, se)
        return pd.DataFrame(
            [{"group": "overall", "estimate": est, "std_error": se, "ci_lower": ci_lo, "ci_upper": ci_hi}]
        )

    if aggregation == "cohort":
        group_col = cohort_col
    elif aggregation == "calendar_time":
        group_col = time_col
    elif aggregation == "event_time":
        group_col = "_rel_time"
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    records = []
    for g_val, grp in df.groupby(group_col):
        est = grp[att_col].mean()
        se = float(np.sqrt(np.mean(grp[se_col] ** 2) / len(grp)))
        ci_lo, ci_hi = _make_ci(est, se)
        records.append({"group": g_val, "estimate": est, "std_error": se, "ci_lower": ci_lo, "ci_upper": ci_hi})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 9. Staggered treatment adoption DiD
# ---------------------------------------------------------------------------


def staggered_did(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treatment_time: str,
    *,
    covariates: list[str] | None = None,
    n_bootstrap: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Staggered DiD via group-time ATTs with aggregation.

    A convenience wrapper around :func:`group_time_att` and
    :func:`aggregate_gt_att` that returns overall, cohort-level, and
    event-time aggregated results.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome, unit, time, treatment_time : str
        Column names.
    covariates : list of str, optional
        Covariates for doubly-robust estimation.
    n_bootstrap : int
        Bootstrap replications.
    seed : int
        Random seed.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Keys: ``group_time`` (pd.DataFrame), ``overall`` (pd.DataFrame),
        ``by_cohort`` (pd.DataFrame), ``by_event_time`` (pd.DataFrame).
    """
    gt = group_time_att(
        data,
        outcome,
        unit,
        time,
        treatment_time,
        covariates=covariates,
        n_bootstrap=n_bootstrap,
        seed=seed,
        alpha=alpha,
    )
    return {
        "group_time": gt,
        "overall": aggregate_gt_att(gt, aggregation="overall"),
        "by_cohort": aggregate_gt_att(gt, aggregation="cohort"),
        "by_event_time": aggregate_gt_att(gt, aggregation="event_time"),
    }


# ---------------------------------------------------------------------------
# 10. Doubly-robust DiD (Sant'Anna & Zhao 2020)
# ---------------------------------------------------------------------------


def did_doubly_robust(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    post: str,
    covariates: list[str],
    *,
    ps_model: str = "logistic",
    or_model: str = "linear",
    cluster: str | None = None,
    n_bootstrap: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> DiDResult:
    r"""Doubly-robust DiD estimator (Sant'Anna & Zhao, 2020).

    Combines an outcome regression model with an inverse-probability
    weighting model.  The estimator is consistent if either model is
    correctly specified.

    .. math::

        \hat\tau_{\text{DR}} = \frac{1}{n_1}\sum_{i:D_i=1}\bigl[\Delta Y_i
        - \hat\mu_0(\mathbf X_i)\bigr]
        - \frac{1}{\sum_{j:D_j=0}\hat w_j}
        \sum_{j:D_j=0}\hat w_j\bigl[\Delta Y_j
        - \hat\mu_0(\mathbf X_j)\bigr]

    Parameters
    ----------
    data : pd.DataFrame
    outcome, treatment, post : str
    covariates : list of str
    ps_model : str
        ``"logistic"`` or ``"gbm"`` for propensity score model.
    or_model : str
        ``"linear"`` or ``"gbm"`` for outcome regression model.
    cluster : str, optional
    n_bootstrap : int
    seed : int
    alpha : float

    Returns
    -------
    DiDResult

    References
    ----------
    Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust
    difference-in-differences estimators. *Journal of Econometrics*,
    219(1), 101--122.
    """
    rng = np.random.default_rng(seed)
    df = data.dropna(subset=[outcome, treatment, post] + covariates).copy()

    d = df[treatment].values.astype(float)
    p = df[post].values.astype(float)
    y = df[outcome].values.astype(float)
    X_cov = df[covariates].values.astype(float)

    def _dr_estimate(d_v, p_v, y_v, X_v):
        # Propensity score
        if ps_model == "gbm":
            ps_fit = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed)
        else:
            ps_fit = LogisticRegression(max_iter=1000, solver="lbfgs", penalty=None)
        ps_fit.fit(X_v, d_v)
        ps_vals = ps_fit.predict_proba(X_v)[:, 1]

        # Outcome regression on controls in post-period
        ctrl_post = (d_v == 0) & (p_v == 1)
        ctrl_pre = (d_v == 0) & (p_v == 0)
        treat_post = (d_v == 1) & (p_v == 1)
        treat_pre = (d_v == 1) & (p_v == 0)

        if or_model == "gbm":
            or_fit_post = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=seed)
            or_fit_pre = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=seed)
        else:
            or_fit_post = LinearRegression()
            or_fit_pre = LinearRegression()

        if ctrl_post.sum() < 2 or ctrl_pre.sum() < 2:
            # Fall back to simple mean difference
            if treat_post.sum() > 0 and treat_pre.sum() > 0:
                return float(
                    (y_v[treat_post].mean() - y_v[treat_pre].mean()) - (y_v[ctrl_post].mean() - y_v[ctrl_pre].mean())
                )
            return 0.0

        or_fit_post.fit(X_v[ctrl_post], y_v[ctrl_post])
        or_fit_pre.fit(X_v[ctrl_pre], y_v[ctrl_pre])

        mu0_post = or_fit_post.predict(X_v)
        mu0_pre = or_fit_pre.predict(X_v)

        # DR ATT
        n1 = d_v.sum()
        if n1 == 0:
            return 0.0

        att_or = float(np.mean(y_v[treat_post] - mu0_post[treat_post]) - np.mean(y_v[treat_pre] - mu0_pre[treat_pre]))

        # IPW correction (DRDID::drdid_rc1): odds weights on controls,
        # controls with p >= 0.995 trimmed
        w = _odds_weights(d_v.astype(int), ps_vals)
        ipw_correction = 0.0
        if (d_v == 0).sum() > 0:
            resid_post = y_v[(d_v == 0) & (p_v == 1)] - mu0_post[(d_v == 0) & (p_v == 1)]
            resid_pre = y_v[(d_v == 0) & (p_v == 0)] - mu0_pre[(d_v == 0) & (p_v == 0)]
            if len(resid_post) > 0 and len(resid_pre) > 0:
                w_post = w[(d_v == 0) & (p_v == 1)]
                w_pre = w[(d_v == 0) & (p_v == 0)]
                ipw_correction = float(
                    np.sum(w_post * resid_post) / max(np.sum(w_post), 1e-10)
                    - np.sum(w_pre * resid_pre) / max(np.sum(w_pre), 1e-10)
                )

        return att_or - ipw_correction

    est = _dr_estimate(d, p, y, X_cov)

    # Bootstrap inference
    n = len(df)
    boot_ests = []
    members = None
    if cluster is not None and cluster in data.columns:
        # cluster bootstrap: whole clusters are resampled
        cl = data.loc[df.index, cluster].values
        cl_ids = np.unique(cl)
        members = {c: np.flatnonzero(cl == c) for c in cl_ids}
    for _ in range(n_bootstrap):
        if members is None:
            idx = rng.choice(n, size=n, replace=True)
        else:
            picked = rng.choice(cl_ids, size=len(cl_ids), replace=True)
            idx = np.concatenate([members[c] for c in picked])
        try:
            b = _dr_estimate(d[idx], p[idx], y[idx], X_cov[idx])
            boot_ests.append(b)
        except Exception:
            continue

    se_est = float(np.std(boot_ests, ddof=1)) if len(boot_ests) > 1 else np.nan
    t_val = est / se_est if se_est > 0 else 0.0
    p_val = float(2 * stats.norm.sf(abs(t_val)))
    ci_lo, ci_hi = _make_ci(est, se_est, alpha)

    return DiDResult(
        estimate=est,
        std_error=se_est,
        t_stat=t_val,
        p_value=p_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_treated=int(d.sum()),
        n_control=int((1 - d).sum()),
        method="did_doubly_robust",
    )


# ---------------------------------------------------------------------------
# 11. Triple Differences (DDD)
# ---------------------------------------------------------------------------


def did_triple_difference(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    post: str,
    third_diff: str,
    *,
    covariates: list[str] | None = None,
    cluster: str | None = None,
    alpha: float = 0.05,
) -> DiDResult:
    r"""Triple-difference (DDD) estimator.

    Extends the standard DiD by adding a third differencing dimension
    (e.g., within-state variation across affected and unaffected
    sub-populations).

    .. math::

        Y = \alpha + \beta_1 D + \beta_2 \text{Post} + \beta_3 S
        + \beta_4 (D \times \text{Post}) + \beta_5 (D \times S)
        + \beta_6 (\text{Post} \times S)
        + \tau (D \times \text{Post} \times S) + \varepsilon

    The DDD estimate is the coefficient on the three-way interaction.

    Parameters
    ----------
    data : pd.DataFrame
    outcome, treatment, post, third_diff : str
        Column names.  *third_diff* is the binary variable for the
        additional differencing group (e.g., age group, sub-population).
    covariates : list of str, optional
    cluster : str, optional
    alpha : float

    Returns
    -------
    DiDResult
    """
    df = data.dropna(subset=[outcome, treatment, post, third_diff]).copy()
    d = df[treatment].values.astype(float)
    p = df[post].values.astype(float)
    s = df[third_diff].values.astype(float)
    y = df[outcome].values.astype(float)

    dp = d * p
    ds = d * s
    ps = p * s
    dps = d * p * s

    parts = [d, p, s, dp, ds, ps, dps]
    if covariates:
        for c in covariates:
            parts.append(df[c].values.astype(float))

    X = _add_intercept(np.column_stack(parts))
    cluster_ids = df[cluster].values if cluster else None
    beta, se = _ols_robust_se(X, y, cluster_ids=cluster_ids)

    # DDD coefficient is at index 7 (intercept + 6 terms before it)
    tau_idx = 7
    est = float(beta[tau_idx])
    se_est = float(se[tau_idx])
    t_val = est / se_est if se_est > 0 else 0.0
    p_val = float(2 * stats.norm.sf(abs(t_val)))
    ci_lo, ci_hi = _make_ci(est, se_est, alpha)

    return DiDResult(
        estimate=est,
        std_error=se_est,
        t_stat=t_val,
        p_value=p_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_treated=int(d.sum()),
        n_control=int((1 - d).sum()),
        method="did_triple_difference",
    )


# ---------------------------------------------------------------------------
# 12. Bacon decomposition
# ---------------------------------------------------------------------------


def bacon_decomposition(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    unit: str,
    time: str,
) -> BaconDecomposition:
    """Goodman-Bacon (2021) decomposition of the TWFE DiD estimator.

    Decomposes the two-way fixed-effects DiD estimate into a weighted
    average of all possible 2x2 DiD comparisons: earlier-vs-later
    treated, treated-vs-never-treated, and later-vs-earlier treated.

    Parameters
    ----------
    data : pd.DataFrame
        Balanced panel data.
    outcome : str
        Outcome column.
    treatment : str
        Binary treatment indicator that turns on at treatment onset.
    unit : str
        Unit identifier.
    time : str
        Time period.

    Returns
    -------
    BaconDecomposition

    References
    ----------
    Goodman-Bacon, A. (2021). Difference-in-differences with variation
    in treatment timing. *Journal of Econometrics*, 225(2), 254--277.
    """
    df = data.copy()
    periods = sorted(df[time].unique())
    T_total = len(periods)

    # Determine treatment timing for each unit
    unit_timing = {}
    for u in df[unit].unique():
        u_data = df[df[unit] == u].sort_values(time)
        treated_periods = u_data[u_data[treatment] == 1][time]
        if len(treated_periods) == 0:
            unit_timing[u] = np.inf  # never-treated
        else:
            unit_timing[u] = treated_periods.min()

    df["_treat_time"] = df[unit].map(unit_timing)

    # Group units by treatment timing
    timing_groups = {}
    for u, t_g in unit_timing.items():
        timing_groups.setdefault(t_g, []).append(u)

    group_keys = sorted([k for k in timing_groups if np.isfinite(k)])
    never_key = np.inf if np.inf in timing_groups else None

    components = []

    def _simple_2x2(units_treat, units_ctrl, pre_periods, post_periods):
        """Compute a simple 2x2 DiD for a subset."""
        df_t_pre = df[(df[unit].isin(units_treat)) & (df[time].isin(pre_periods))]
        df_t_post = df[(df[unit].isin(units_treat)) & (df[time].isin(post_periods))]
        df_c_pre = df[(df[unit].isin(units_ctrl)) & (df[time].isin(pre_periods))]
        df_c_post = df[(df[unit].isin(units_ctrl)) & (df[time].isin(post_periods))]

        if any(len(x) == 0 for x in [df_t_pre, df_t_post, df_c_pre, df_c_post]):
            return None

        est = (df_t_post[outcome].mean() - df_t_pre[outcome].mean()) - (
            df_c_post[outcome].mean() - df_c_pre[outcome].mean()
        )
        return float(est)

    # All pairwise comparisons of treatment-timing groups
    for g_early, g_late in combinations(group_keys, 2):
        # Early-vs-late: early treated vs late treated (using pre-late period)
        pre_pds = [p for p in periods if p < g_early]
        mid_pds = [p for p in periods if g_early <= p < g_late]

        if pre_pds and mid_pds:
            est = _simple_2x2(timing_groups[g_early], timing_groups[g_late], pre_pds, mid_pds)
            if est is not None:
                n_e = len(timing_groups[g_early])
                n_l = len(timing_groups[g_late])
                # Weight proportional to subsample size * share of variance
                weight = (n_e * n_l * len(mid_pds) * len(pre_pds)) / (T_total**2)
                components.append(
                    {
                        "group1": g_early,
                        "group2": g_late,
                        "estimate": est,
                        "weight": weight,
                        "type": "earlier_vs_later",
                    }
                )

        # Late-vs-early (reverse comparison, using post-late period)
        post_pds = [p for p in periods if p >= g_late]
        if mid_pds and post_pds:
            est = _simple_2x2(timing_groups[g_late], timing_groups[g_early], mid_pds, post_pds)
            if est is not None:
                n_e = len(timing_groups[g_early])
                n_l = len(timing_groups[g_late])
                weight = (n_e * n_l * len(post_pds) * len(mid_pds)) / (T_total**2)
                components.append(
                    {
                        "group1": g_late,
                        "group2": g_early,
                        "estimate": est,
                        "weight": weight,
                        "type": "later_vs_earlier",
                    }
                )

    # Treated vs never-treated
    if never_key is not None:
        for g in group_keys:
            pre_pds = [p for p in periods if p < g]
            post_pds = [p for p in periods if p >= g]
            if pre_pds and post_pds:
                est = _simple_2x2(timing_groups[g], timing_groups[never_key], pre_pds, post_pds)
                if est is not None:
                    n_t = len(timing_groups[g])
                    n_c = len(timing_groups[never_key])
                    weight = (n_t * n_c * len(post_pds) * len(pre_pds)) / (T_total**2)
                    components.append(
                        {
                            "group1": g,
                            "group2": "never_treated",
                            "estimate": est,
                            "weight": weight,
                            "type": "treated_vs_never",
                        }
                    )

    comp_df = pd.DataFrame(components)
    if len(comp_df) > 0:
        comp_df["weight"] = comp_df["weight"] / comp_df["weight"].sum()
        overall = float((comp_df["estimate"] * comp_df["weight"]).sum())
    else:
        overall = np.nan

    return BaconDecomposition(components=comp_df, overall_estimate=overall)


# ---------------------------------------------------------------------------
# 13. Synthetic DiD (Arkhangelsky et al., 2021)
# ---------------------------------------------------------------------------


def _sdid_fw(Ym, zeta, lam0, min_decrease, max_iter):
    """synthdid:::sc.weight.fw with intercept: Frank-Wolfe with exact line
    search on the simplex; stops when the objective falls by less than
    min_decrease^2."""
    N0 = len(Ym)
    T0 = len(Ym[0]) - 1
    cm = [sum(r[j] for r in Ym) / N0 for j in range(T0 + 1)]
    Y = [[r[j] - cm[j] for j in range(T0 + 1)] for r in Ym]
    A = [r[:T0] for r in Y]
    b = [r[T0] for r in Y]
    lam = [1.0 / T0] * T0 if lam0 is None else list(lam0)
    eta = N0 * zeta**2
    vals = []
    t = 0
    while t < max_iter and (t < 2 or vals[t - 2] - vals[t - 1] > min_decrease**2):
        t += 1
        Ax = [sum(A[i][j] * lam[j] for j in range(T0)) for i in range(N0)]
        hg = [sum(A[i][j] * (Ax[i] - b[i]) for i in range(N0)) + eta * lam[j] for j in range(T0)]
        k = min(range(T0), key=lambda j: hg[j])
        dx = [-v for v in lam]
        dx[k] = 1 - lam[k]
        if any(v != 0 for v in dx):
            derr = [A[i][k] - Ax[i] for i in range(N0)]
            step = -sum(h * d for h, d in zip(hg, dx)) / (sum(e * e for e in derr) + eta * sum(d * d for d in dx))
            st = min(1.0, max(0.0, step))
            lam = [lam[j] + st * dx[j] for j in range(T0)]
        err = [sum(Y[i][j] * lam[j] for j in range(T0)) - Y[i][T0] for i in range(N0)]
        vals.append(zeta**2 * sum(v * v for v in lam) + sum(e * e for e in err) / N0)
    return lam


def _sdid_sparsify(v):
    m = max(v) / 4
    w = [x if x > m else 0.0 for x in v]
    tot = sum(w)
    return [x / tot for x in w]


def _sdid_core(Y, N0, T0, opts, omega=None, lam=None, update_omega=True, update_lambda=True):
    """synthdid::synthdid_estimate without covariates (Arkhangelsky, Athey,
    Hirshberg, Imbens & Wager 2021)."""
    N, T = len(Y), len(Y[0])
    N1, T1 = N - N0, T - T0
    Yc = [list(Y[i][:T0]) + [sum(Y[i][T0:]) / T1] for i in range(N0)]
    Yc.append(
        [sum(Y[i][j] for i in range(N0, N)) / N1 for j in range(T0)]
        + [sum(Y[i][j] for i in range(N0, N) for j in range(T0, T)) / (N1 * T1)]
    )
    if update_lambda:
        l1 = _sdid_fw(Yc[:N0], opts["zeta_lambda"], lam, opts["min_decrease"], 100)
        lam = _sdid_fw(Yc[:N0], opts["zeta_lambda"], _sdid_sparsify(l1), opts["min_decrease"], 10000)
    if update_omega:
        Yo = [[Yc[i][j] for i in range(N0 + 1)] for j in range(T0)]
        o1 = _sdid_fw(Yo, opts["zeta_omega"], omega, opts["min_decrease"], 100)
        omega = _sdid_fw(Yo, opts["zeta_omega"], _sdid_sparsify(o1), opts["min_decrease"], 10000)
    wu = [-v for v in omega] + [1.0 / N1] * N1
    wt = [-v for v in lam] + [1.0 / T1] * T1
    tau = sum(wu[i] * sum(Y[i][j] * wt[j] for j in range(T)) for i in range(N))
    return tau, omega, lam


def synthetic_did(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treatment_time: str,
    *,
    treated_units: list | None = None,
    zeta: float | None = None,
    n_bootstrap: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
    se_method: str = "bootstrap",
) -> DiDResult:
    """Synthetic difference-in-differences (Arkhangelsky, Athey, Hirshberg,
    Imbens & Wager 2021), computed as ``synthdid::synthdid_estimate``: unit
    and time weights by Frank-Wolfe with a sparsify pass, zeta_omega =
    (N1 T1)^(1/4) sigma and zeta_lambda = 1e-6 sigma, sigma the sd of the
    controls' pre-period first differences. Block design: every treated
    unit starts at the first treatment time.

    Parameters
    ----------
    data : pd.DataFrame
        Long panel.
    outcome, unit, time : str
    treatment_time : str
        Column of first treatment time (NaN or 0 for never treated).
    treated_units : list, optional
    zeta : float, optional
        Overrides zeta_omega.
    n_bootstrap : int
        Replications for the bootstrap and placebo standard errors.
    seed : int
    alpha : float
    se_method : {"bootstrap", "jackknife", "placebo"}
        As ``synthdid::vcov``: the jackknife holds the weights fixed; the
        bootstrap and placebo re-solve them from the renormalised
        full-sample weights.

    Returns
    -------
    DiDResult
    """
    df = data.copy()
    tt = [float(v) for v in df[treatment_time].tolist()]
    units = df[unit].tolist()
    if treated_units is None:
        treated_units = sorted({u for u, g in zip(units, tt) if g == g and g > 0})
    if not treated_units:
        raise ValueError("No treated units found.")
    first_treat = min(g for u, g in zip(units, tt) if u in set(treated_units) and g == g)
    control_units = sorted(set(units) - set(treated_units))
    times = sorted(set(df[time].tolist()))
    pre = [t for t in times if t < first_treat]
    if len(pre) < 2 or len(pre) == len(times):
        raise ValueError("Need at least two pre periods and one post period.")
    cell = {}
    for u, t, yv in zip(units, df[time].tolist(), df[outcome].tolist()):
        cell[(u, t)] = float(yv)
    Y = [[cell[(u, t)] for t in times] for u in control_units + list(treated_units)]
    N0, T0 = len(control_units), len(pre)
    N1, T1 = len(Y) - N0, len(times) - T0
    diffs = [Y[i][j + 1] - Y[i][j] for i in range(N0) for j in range(T0 - 1)]
    mu = sum(diffs) / len(diffs)
    noise = math.sqrt(sum((v - mu) ** 2 for v in diffs) / (len(diffs) - 1))
    opts = {
        "zeta_omega": ((N1 * T1) ** 0.25) * noise if zeta is None else float(zeta),
        "zeta_lambda": 1e-6 * noise,
        "min_decrease": 1e-5 * noise,
    }
    tau, omega, lam = _sdid_core(Y, N0, T0, opts)

    def norm1(v):
        tot = sum(v)
        return [x / tot for x in v] if tot > 0 else [1.0 / len(v)] * len(v)

    rng = np.random.default_rng(seed)
    se_est = float("nan")
    N = len(Y)
    if se_method == "jackknife":
        if N0 < N - 1 and sum(1 for v in omega if v != 0) > 1:
            jk = []
            for i in range(N):
                ind = [k for k in range(N) if k != i]
                n0 = sum(1 for k in ind if k < N0)
                jk.append(
                    _sdid_core(
                        [Y[k] for k in ind],
                        n0,
                        T0,
                        opts,
                        omega=norm1([omega[k] for k in ind if k < N0]),
                        lam=lam,
                        update_omega=False,
                        update_lambda=False,
                    )[0]
                )
            m = sum(jk) / N
            se_est = math.sqrt((N - 1) / N * sum((v - m) ** 2 for v in jk))
    elif se_method == "placebo":
        if N0 > N1:
            reps = []
            for _ in range(n_bootstrap):
                ind = [int(v) for v in rng.permutation(N0).tolist()]
                n0 = N0 - N1
                reps.append(
                    _sdid_core([Y[k] for k in ind], n0, T0, opts, omega=norm1([omega[k] for k in ind[:n0]]), lam=lam)[0]
                )
            m = sum(reps) / len(reps)
            se_est = math.sqrt(sum((v - m) ** 2 for v in reps) / len(reps))
    elif se_method == "bootstrap":
        if N0 < N - 1:
            reps = []
            while len(reps) < n_bootstrap:
                ind = sorted(int(v) for v in rng.integers(0, N, size=N).tolist())
                if all(k < N0 for k in ind) or all(k >= N0 for k in ind):
                    continue
                reps.append(
                    _sdid_core(
                        [Y[k] for k in ind],
                        sum(1 for k in ind if k < N0),
                        T0,
                        opts,
                        omega=norm1([omega[k] for k in ind if k < N0]),
                        lam=lam,
                    )[0]
                )
            m = sum(reps) / len(reps)
            se_est = math.sqrt(sum((v - m) ** 2 for v in reps) / len(reps))
    else:
        raise ValueError("se_method must be 'bootstrap', 'jackknife' or 'placebo'")
    t_val = tau / se_est if se_est > 0 else 0.0
    p_val = float(2 * stats.norm.sf(abs(t_val))) if se_est == se_est else float("nan")
    ci_lo, ci_hi = _make_ci(tau, se_est, alpha)
    return DiDResult(
        estimate=float(tau),
        std_error=se_est,
        t_stat=t_val,
        p_value=p_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_treated=N1,
        n_control=N0,
        method="synthetic_did",
        details={
            "unit_weights": dict(zip(control_units, omega)),
            "time_weights": dict(zip(pre, lam)),
            "zeta": opts["zeta_omega"],
            "se_method": se_method,
        },
    )


# ---------------------------------------------------------------------------
# 14. Cluster-robust and wild cluster bootstrap
# ---------------------------------------------------------------------------


def wild_cluster_bootstrap(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    post: str,
    cluster: str,
    *,
    covariates: list[str] | None = None,
    n_bootstrap: int = 999,
    weight_type: str = "rademacher",
    seed: int = 42,
    alpha: float = 0.05,
) -> DiDResult:
    """DiD with wild cluster bootstrap p-values (Cameron, Gelbach & Miller, 2008).

    Recommended when the number of clusters is small (< 50).  Multiplies
    cluster-level residuals by random weights drawn from the Rademacher
    or Webb distributions.

    Parameters
    ----------
    data : pd.DataFrame
    outcome, treatment, post, cluster : str
    covariates : list of str, optional
    n_bootstrap : int
        Number of bootstrap replications.
    weight_type : str
        ``"rademacher"`` (+1/-1 with equal probability) or ``"webb"``
        (six-point distribution).
    seed : int
    alpha : float

    Returns
    -------
    DiDResult
        The *p_value* is the bootstrap p-value.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based
    improvements for inference with clustered errors. *Review of Economics
    and Statistics*, 90(3), 414--427.
    """
    rng = np.random.default_rng(seed)
    df = data.dropna(subset=[outcome, treatment, post, cluster]).copy()

    d = df[treatment].values.astype(float)
    p = df[post].values.astype(float)
    y = df[outcome].values.astype(float)
    interaction = d * p

    if covariates:
        X_cov = df[covariates].values.astype(float)
        X = _add_intercept(np.column_stack([d, p, interaction, X_cov]))
    else:
        X = _add_intercept(np.column_stack([d, p, interaction]))

    cluster_ids = df[cluster].values
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)

    # Full-sample estimate
    beta_full, se_full = _ols_robust_se(X, y, cluster_ids=cluster_ids)
    tau_idx = 3
    t_stat_full = beta_full[tau_idx] / se_full[tau_idx] if se_full[tau_idx] > 0 else 0.0

    # Restricted residuals (impose null: coefficient on interaction = 0)
    X_r = np.delete(X, tau_idx, axis=1)
    beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
    resid_r = y - X_r @ beta_r

    # Webb 6-point distribution
    webb_vals = np.array(
        [-np.sqrt(3 / 2), -np.sqrt(2 / 2), -np.sqrt(1 / 2), np.sqrt(1 / 2), np.sqrt(2 / 2), np.sqrt(3 / 2)]
    )

    # Rademacher weights are enumerated in full when 2^G <= B, as
    # fwildclusterboot::boottest and Stata's boottest do
    enum = weight_type != "webb" and n_bootstrap >= 2**G
    draws = product([-1.0, 1.0], repeat=G) if enum else range(n_bootstrap)
    boot_t_stats = []
    for draw in draws:
        # Draw cluster-level weights
        if enum:
            w = np.array(draw)
        else:
            w = rng.choice(webb_vals, size=G) if weight_type == "webb" else rng.choice([-1.0, 1.0], size=G)

        # Construct bootstrap outcome
        y_star = X_r @ beta_r  # fitted under null
        for i, c in enumerate(unique_clusters):
            mask = cluster_ids == c
            y_star[mask] += w[i] * resid_r[mask]

        beta_b, se_b = _ols_robust_se(X, y_star, cluster_ids=cluster_ids)
        t_b = beta_b[tau_idx] / se_b[tau_idx] if se_b[tau_idx] > 0 else 0.0
        boot_t_stats.append(abs(t_b))

    # Bootstrap p-value
    # symmetric p-value, strict inequality (boottest); the draws w = +-1
    # reproduce t itself and must not count
    boot_p = float(np.mean(np.array(boot_t_stats) > abs(t_stat_full) * (1 + 1e-10)))

    est = float(beta_full[tau_idx])
    se_est = float(se_full[tau_idx])
    ci_lo, ci_hi = _make_ci(est, se_est, alpha)

    return DiDResult(
        estimate=est,
        std_error=se_est,
        t_stat=float(t_stat_full),
        p_value=boot_p,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_treated=int(d.sum()),
        n_control=int((1 - d).sum()),
        method="wild_cluster_bootstrap",
        details={
            "n_clusters": G,
            "n_bootstrap": len(boot_t_stats),
            "weight_type": weight_type,
            "full_enumeration": enum,
        },
    )


# ---------------------------------------------------------------------------
# 15. DiD with continuous treatment
# ---------------------------------------------------------------------------


def did_continuous_treatment(
    data: pd.DataFrame,
    outcome: str,
    dose: str,
    post: str,
    *,
    covariates: list[str] | None = None,
    cluster: str | None = None,
    alpha: float = 0.05,
) -> DiDResult:
    r"""DiD with a continuous treatment variable (dose--response DiD).

    Models the outcome as:

    .. math::

        Y_{it} = \alpha + \beta\,\text{Dose}_i + \gamma\,\text{Post}_t
        + \tau\,(\text{Dose}_i \times \text{Post}_t)
        + X_{it}'\delta + \varepsilon_{it}

    :math:`\hat\tau` is the marginal effect of a one-unit increase in
    treatment intensity in the post period.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    dose : str
        Continuous treatment intensity column.
    post : str
        Binary post-period indicator.
    covariates : list of str, optional
    cluster : str, optional
    alpha : float

    Returns
    -------
    DiDResult
    """
    df = data.dropna(subset=[outcome, dose, post]).copy()
    d = df[dose].values.astype(float)
    p = df[post].values.astype(float)
    y = df[outcome].values.astype(float)
    interaction = d * p

    parts = [d, p, interaction]
    if covariates:
        for c in covariates:
            parts.append(df[c].values.astype(float))

    X = _add_intercept(np.column_stack(parts))
    cluster_ids = df[cluster].values if cluster else None
    beta, se = _ols_robust_se(X, y, cluster_ids=cluster_ids)

    tau_idx = 3
    est = float(beta[tau_idx])
    se_est = float(se[tau_idx])
    t_val = est / se_est if se_est > 0 else 0.0
    p_val = float(2 * stats.norm.sf(abs(t_val)))
    ci_lo, ci_hi = _make_ci(est, se_est, alpha)

    return DiDResult(
        estimate=est,
        std_error=se_est,
        t_stat=t_val,
        p_value=p_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_treated=int((d > 0).sum()),
        n_control=int((d == 0).sum()),
        method="did_continuous_treatment",
    )


# ---------------------------------------------------------------------------
# 16. Fuzzy DiD
# ---------------------------------------------------------------------------


def did_fuzzy(
    data: pd.DataFrame,
    outcome: str,
    assignment: str,
    takeup: str,
    post: str,
    *,
    covariates: list[str] | None = None,
    cluster: str | None = None,
    alpha: float = 0.05,
) -> DiDResult:
    r"""Fuzzy DiD estimator for settings with imperfect compliance.

    Uses the interaction of *assignment* :math:`\times` *post* as an
    instrument for *takeup* :math:`\times` *post* in a 2SLS
    framework to recover a local average treatment effect (LATE).

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    assignment : str
        Intent-to-treat assignment indicator (binary).
    takeup : str
        Actual treatment takeup indicator (binary).
    post : str
        Post-period indicator.
    covariates : list of str, optional
    cluster : str, optional
    alpha : float

    Returns
    -------
    DiDResult
        The ``estimate`` is the LATE from fuzzy DiD.
    """
    df = data.dropna(subset=[outcome, assignment, takeup, post]).copy()
    z = df[assignment].values.astype(float)
    d = df[takeup].values.astype(float)
    p = df[post].values.astype(float)
    y = df[outcome].values.astype(float)

    # 2SLS: instrument = Z * Post, endogenous = D * Post
    zp = z * p
    dp = d * p

    exog = [z, p, d]
    if covariates:
        for c in covariates:
            exog.append(df[c].values.astype(float))

    X_exog = _add_intercept(np.column_stack(exog))

    # First stage: D*Post = pi0 + pi1*(Z*Post) + exog + e
    X_first = np.column_stack([X_exog, zp])
    beta_first = np.linalg.lstsq(X_first, dp, rcond=None)[0]
    dp_hat = X_first @ beta_first

    # Second stage: Y = beta0 + tau*(D*Post_hat) + exog + u
    X_second = np.column_stack([X_exog, dp_hat])
    cluster_ids = df[cluster].values if cluster else None
    beta_2, se_2 = _ols_robust_se(X_second, y, cluster_ids=cluster_ids)

    tau_idx = X_second.shape[1] - 1  # last column
    est = float(beta_2[tau_idx])
    se_est = float(se_2[tau_idx])
    t_val = est / se_est if se_est > 0 else 0.0
    p_val = float(2 * stats.norm.sf(abs(t_val)))
    ci_lo, ci_hi = _make_ci(est, se_est, alpha)

    # First-stage F-statistic
    resid_r = dp - X_exog @ np.linalg.lstsq(X_exog, dp, rcond=None)[0]
    resid_u = dp - X_first @ beta_first
    ssr_r = float(np.sum(resid_r**2))
    ssr_u = float(np.sum(resid_u**2))
    n = len(y)
    k = X_first.shape[1]
    f_stat = ((ssr_r - ssr_u) / 1) / (ssr_u / (n - k)) if ssr_u > 0 else 0.0

    return DiDResult(
        estimate=est,
        std_error=se_est,
        t_stat=t_val,
        p_value=p_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_treated=int(d.sum()),
        n_control=int((1 - d).sum()),
        method="did_fuzzy",
        details={"first_stage_f": f_stat, "compliance_rate": float(d.mean())},
    )


# ---------------------------------------------------------------------------
# 17. Placebo and falsification tests
# ---------------------------------------------------------------------------


def placebo_test_time(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    time: str,
    true_treatment_time: Any,
    placebo_times: list[Any],
    *,
    covariates: list[str] | None = None,
    cluster: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run placebo DiD tests at fake treatment times.

    For each time in *placebo_times*, re-define the post indicator as
    ``time >= placebo_time`` using only pre-treatment data (before
    *true_treatment_time*) and estimate a DiD.  A well-identified design
    should yield estimates close to zero.

    Parameters
    ----------
    data : pd.DataFrame
    outcome, treatment, time : str
    true_treatment_time : any
        The actual treatment onset time (to restrict to pre-period data).
    placebo_times : list
        Candidate fake treatment times to test.
    covariates : list of str, optional
    cluster : str, optional
    alpha : float

    Returns
    -------
    pd.DataFrame
        One row per placebo time with ``placebo_time``, ``estimate``,
        ``std_error``, ``p_value``, ``significant``.
    """
    df_pre = data[data[time] < true_treatment_time].copy()
    results = []

    for pt in placebo_times:
        df_test = df_pre.copy()
        df_test["_placebo_post"] = (df_test[time] >= pt).astype(int)

        if df_test["_placebo_post"].nunique() < 2:
            continue

        res = did_2x2(
            df_test,
            outcome,
            treatment,
            "_placebo_post",
            covariates=covariates,
            cluster=cluster,
            alpha=alpha,
        )
        results.append(
            {
                "placebo_time": pt,
                "estimate": res.estimate,
                "std_error": res.std_error,
                "p_value": res.p_value,
                "significant": res.p_value < alpha,
            }
        )

    return pd.DataFrame(results)


def placebo_test_outcome(
    data: pd.DataFrame,
    placebo_outcomes: list[str],
    treatment: str,
    post: str,
    *,
    covariates: list[str] | None = None,
    cluster: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run placebo DiD on outcomes that should not be affected by treatment.

    Parameters
    ----------
    data : pd.DataFrame
    placebo_outcomes : list of str
        Outcome columns expected to show no treatment effect.
    treatment, post : str
    covariates : list of str, optional
    cluster : str, optional
    alpha : float

    Returns
    -------
    pd.DataFrame
        One row per placebo outcome.
    """
    results = []
    for out in placebo_outcomes:
        if out not in data.columns:
            continue
        res = did_2x2(
            data,
            out,
            treatment,
            post,
            covariates=covariates,
            cluster=cluster,
            alpha=alpha,
        )
        results.append(
            {
                "outcome": out,
                "estimate": res.estimate,
                "std_error": res.std_error,
                "p_value": res.p_value,
                "significant": res.p_value < alpha,
            }
        )
    return pd.DataFrame(results)


def placebo_test_group(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    post: str,
    group_col: str,
    unaffected_groups: list[Any],
    *,
    covariates: list[str] | None = None,
    cluster: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run placebo DiD on groups that should not be affected.

    Parameters
    ----------
    data : pd.DataFrame
    outcome, treatment, post : str
    group_col : str
        Column defining sub-groups.
    unaffected_groups : list
        Group values where no treatment effect is expected.
    covariates : list of str, optional
    cluster : str, optional
    alpha : float

    Returns
    -------
    pd.DataFrame
    """
    results = []
    for g in unaffected_groups:
        df_g = data[data[group_col] == g].copy()
        if df_g[treatment].nunique() < 2:
            continue
        res = did_2x2(
            df_g,
            outcome,
            treatment,
            post,
            covariates=covariates,
            cluster=cluster,
            alpha=alpha,
        )
        results.append(
            {
                "group": g,
                "estimate": res.estimate,
                "std_error": res.std_error,
                "p_value": res.p_value,
                "significant": res.p_value < alpha,
            }
        )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 18. Heterogeneity-robust DiD
# ---------------------------------------------------------------------------


def did_heterogeneous(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    post: str,
    moderator: str,
    *,
    covariates: list[str] | None = None,
    cluster: str | None = None,
    n_quantiles: int = 4,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Heterogeneity-robust DiD: estimate treatment effects by subgroups.

    Splits the sample by quantiles (or categories) of *moderator* and
    estimates separate DiD effects for each stratum.

    Parameters
    ----------
    data : pd.DataFrame
    outcome, treatment, post, moderator : str
    covariates : list of str, optional
    cluster : str, optional
    n_quantiles : int
        Number of quantile bins if *moderator* is continuous.
    alpha : float

    Returns
    -------
    pd.DataFrame
        Columns: ``group``, ``estimate``, ``std_error``, ``ci_lower``,
        ``ci_upper``, ``p_value``, ``n``.
    """
    df = data.copy()

    if pd.api.types.is_numeric_dtype(df[moderator]) and df[moderator].nunique() > n_quantiles:
        df["_mod_group"] = pd.qcut(df[moderator], n_quantiles, labels=False, duplicates="drop")
    else:
        df["_mod_group"] = df[moderator]

    results = []
    for g_val, grp in df.groupby("_mod_group"):
        if grp[treatment].nunique() < 2 or grp[post].nunique() < 2:
            continue
        res = did_2x2(
            grp,
            outcome,
            treatment,
            post,
            covariates=covariates,
            cluster=cluster,
            alpha=alpha,
        )
        results.append(
            {
                "group": g_val,
                "estimate": res.estimate,
                "std_error": res.std_error,
                "ci_lower": res.ci_lower,
                "ci_upper": res.ci_upper,
                "p_value": res.p_value,
                "n": len(grp),
            }
        )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 19. Heterogeneity-robust TWFE (de Chaisemartin & D'Haultfoeuille)
# ---------------------------------------------------------------------------


def did_chaisemartin_dhaultfoeuille(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    unit: str,
    time: str,
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> DiDResult:
    r"""Heterogeneity-robust DiD (de Chaisemartin & D'Haultfoeuille, 2020).

    Computes the instantaneous treatment effect for switchers--units
    whose treatment status changes--using appropriate comparisons.

    .. math::

        \hat\delta = \sum_{(i,t): D_{it}>D_{i,t-1}} w_{it}
        \bigl[\Delta Y_{it} - \Delta \bar Y_{ct}\bigr]

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome, treatment, unit, time : str
    n_bootstrap : int
    seed : int
    alpha : float

    Returns
    -------
    DiDResult

    References
    ----------
    de Chaisemartin, C., & D'Haultfoeuille, X. (2020). Two-way fixed
    effects estimators with heterogeneous treatment effects. *American
    Economic Review*, 110(9), 2964--2996.
    """
    rng = np.random.default_rng(seed)
    df = data.sort_values([unit, time]).copy()
    periods = sorted(df[time].unique())

    estimates = []
    weights = []

    for t_idx in range(1, len(periods)):
        t_cur = periods[t_idx]
        t_prev = periods[t_idx - 1]

        df_cur = df[df[time] == t_cur].set_index(unit)
        df_prev = df[df[time] == t_prev].set_index(unit)
        common_units = df_cur.index.intersection(df_prev.index)

        if len(common_units) == 0:
            continue

        d_cur = df_cur.loc[common_units, treatment].values.astype(float)
        d_prev = df_prev.loc[common_units, treatment].values.astype(float)
        y_cur = df_cur.loc[common_units, outcome].values.astype(float)
        y_prev = df_prev.loc[common_units, outcome].values.astype(float)

        # Switchers: units that went from untreated to treated
        switchers = (d_cur == 1) & (d_prev == 0)
        # Non-switchers staying untreated
        controls = (d_cur == 0) & (d_prev == 0)

        n_switch = switchers.sum()
        n_ctrl = controls.sum()

        if n_switch == 0 or n_ctrl == 0:
            continue

        delta_y_switch = (y_cur[switchers] - y_prev[switchers]).mean()
        delta_y_ctrl = (y_cur[controls] - y_prev[controls]).mean()
        est_t = delta_y_switch - delta_y_ctrl

        estimates.append(est_t)
        weights.append(n_switch)

    if len(estimates) == 0:
        return DiDResult(
            estimate=np.nan,
            std_error=np.nan,
            t_stat=np.nan,
            p_value=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            n_treated=0,
            n_control=0,
            method="chaisemartin_dhaultfoeuille",
        )

    w = np.array(weights, dtype=float)
    w = w / w.sum()
    delta_hat = float(np.sum(w * np.array(estimates)))

    # Bootstrap SE
    units = df[unit].unique()
    boot_ests = []
    for _ in range(n_bootstrap):
        b_units = rng.choice(units, size=len(units), replace=True)
        df_b = pd.concat(
            [df[df[unit] == u].assign(**{unit: f"{u}_{j}"}) for j, u in enumerate(b_units)], ignore_index=True
        )
        try:
            r = did_chaisemartin_dhaultfoeuille(
                df_b,
                outcome,
                treatment,
                unit,
                time,
                n_bootstrap=0,
                seed=seed,
            )
            boot_ests.append(r.estimate)
        except Exception:
            continue

    se_est = float(np.std(boot_ests, ddof=1)) if len(boot_ests) > 1 else np.nan
    t_val = delta_hat / se_est if se_est > 0 else 0.0
    p_val = float(2 * stats.norm.sf(abs(t_val)))
    ci_lo, ci_hi = _make_ci(delta_hat, se_est, alpha)

    return DiDResult(
        estimate=delta_hat,
        std_error=se_est,
        t_stat=t_val,
        p_value=p_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_treated=int(sum(weights)),
        n_control=len(df[unit].unique()) - int(sum(weights)),
        method="chaisemartin_dhaultfoeuille",
    )


# ---------------------------------------------------------------------------
# 20. Sensitivity analysis for DiD
# ---------------------------------------------------------------------------


def did_sensitivity_analysis(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    post: str,
    *,
    covariates: list[str] | None = None,
    delta_range: Sequence[float] | None = None,
    cluster: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    r"""Sensitivity of DiD estimate to violations of parallel trends.

    Following Rambachan & Roth (2023), computes the identified set for
    the ATT under bounded deviations :math:`\delta` from parallel trends:

    .. math::

        |\text{bias}| \le \delta \cdot \hat\sigma

    For each :math:`\delta`, computes a bias-adjusted confidence set.

    Parameters
    ----------
    data : pd.DataFrame
    outcome, treatment, post : str
    covariates : list of str, optional
    delta_range : sequence of float, optional
        Values of :math:`\delta` to evaluate (default: 0 to 2 in 0.25 steps).
    cluster : str, optional
    alpha : float

    Returns
    -------
    pd.DataFrame
        Columns: ``delta``, ``ci_lower``, ``ci_upper``, ``covers_zero``.

    References
    ----------
    Rambachan, A., & Roth, J. (2023). A more credible approach to
    parallel trends. *Review of Economic Studies*, 90(5), 2555--2591.
    """
    if delta_range is None:
        delta_range = np.arange(0, 2.25, 0.25)

    res = did_2x2(data, outcome, treatment, post, covariates=covariates, cluster=cluster, alpha=alpha)

    results = []
    for delta in delta_range:
        bias_bound = delta * res.std_error
        ci_lo = res.estimate - bias_bound - stats.norm.ppf(1 - alpha / 2) * res.std_error
        ci_hi = res.estimate + bias_bound + stats.norm.ppf(1 - alpha / 2) * res.std_error
        results.append(
            {
                "delta": delta,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "covers_zero": ci_lo <= 0 <= ci_hi,
            }
        )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 21. DiD summary diagnostics
# ---------------------------------------------------------------------------


def did_diagnostics(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    post: str,
    *,
    covariates: list[str] | None = None,
    cluster: str | None = None,
) -> dict[str, Any]:
    """Comprehensive diagnostics for a 2x2 DiD setting.

    Checks:
    - Sample sizes by group and period
    - Baseline covariate balance
    - Outcome distributions by group and period
    - Pre-period outcome correlation between groups

    Parameters
    ----------
    data : pd.DataFrame
    outcome, treatment, post : str
    covariates : list of str, optional
    cluster : str, optional

    Returns
    -------
    dict
        Keys: ``sample_sizes``, ``outcome_stats``, ``covariate_balance``.
    """
    df = data.dropna(subset=[outcome, treatment, post]).copy()

    # Sample sizes
    sizes = df.groupby([treatment, post]).size().unstack(fill_value=0)

    # Outcome statistics
    outcome_stats = df.groupby([treatment, post])[outcome].agg(["mean", "std", "median", "min", "max", "count"])

    # Covariate balance (pre-period)
    cov_balance = None
    if covariates:
        df_pre = df[df[post] == 0]
        records = []
        for c in covariates:
            if c not in df_pre.columns:
                continue
            treat_vals = df_pre.loc[df_pre[treatment] == 1, c].astype(float)
            ctrl_vals = df_pre.loc[df_pre[treatment] == 0, c].astype(float)
            mean_diff = float(treat_vals.mean() - ctrl_vals.mean())
            pooled_sd = float(np.sqrt((treat_vals.var(ddof=1) + ctrl_vals.var(ddof=1)) / 2))
            smd = mean_diff / pooled_sd if pooled_sd > 0 else np.nan
            records.append(
                {
                    "covariate": c,
                    "mean_treated": float(treat_vals.mean()),
                    "mean_control": float(ctrl_vals.mean()),
                    "smd": smd,
                }
            )
        cov_balance = pd.DataFrame(records)

    cluster_counts = None
    if cluster and cluster in df.columns:
        cluster_counts = df.groupby([treatment, post])[cluster].nunique().unstack(fill_value=0)

    return {
        "sample_sizes": sizes,
        "outcome_stats": outcome_stats,
        "covariate_balance": cov_balance,
        "cluster_counts": cluster_counts,
    }
