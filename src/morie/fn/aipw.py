# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""
Augmented Inverse Probability Weighting (AIPW) doubly-robust estimator.

Implements ``estimate_aipw`` -- estimates the ATE with double robustness:
consistent if either the propensity score model or outcome model is correct.
"""

from __future__ import annotations

from typing import Any

from . import _array_core as np
from . import _frame_core as pd

class _MissingDep:
    """Placeholder for a dependency being nativized (task #141)."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        raise ImportError(
            "%s is no longer bundled; this code path awaits its native "
            "morie implementation" % self._name)

    def __call__(self, *a, **k):
        raise ImportError(
            "%s is no longer bundled; this code path awaits its native "
            "morie implementation" % self._name)

try:
    from ._ml_core import LinearRegression, LogisticRegression
except ImportError:
    LinearRegression = _MissingDep('LinearRegression')
    LogisticRegression = _MissingDep('LogisticRegression')
try:
    from ._ml_core import LabelEncoder, StandardScaler
except ImportError:
    LabelEncoder = _MissingDep('LabelEncoder')
    StandardScaler = _MissingDep('StandardScaler')

from morie.fn.ps_fit import compute_propensity_scores
from morie.fn.ps_fit import (_ps_design, _ps_irls_beta,
                             _ps_solve)
import math as _math


_PS_EPS = 1e-6


def _ps_keep(ps, trim=(0.1, 0.9), trim_type="value"):
    """Mask of units SURVIVING trimming; only "discard" drops any.

    The discard route is Crump, Hotz, Imbens and Mitnik (2009),
    Biometrika 96(1), 187-199, verified verbatim: "drop all units with
    estimated propensity scores outside the range [0.1,0.9]".

    Discarding CHANGES THE ESTIMAND -- the result is the ATE on the
    retained subpopulation, not on the whole sample.  Callers report
    n_discarded and an estimand note when this route is used.
    """
    if trim_type != "discard" or trim is None:
        return [True] * len(ps)
    lo, hi = float(trim[0]), float(trim[1])
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("trim must satisfy 0 <= lo < hi <= 1")
    return [bool(lo <= float(u) <= hi) for u in ps]


def _trim_ps(ps, trim, trim_type="value"):
    """Trim propensity scores; see :func:`estimate_aipw` for the routes.

    Mirrored exactly by ``.mor_trim_ps`` in the R arm.
    """
    ps = np.asarray(ps, dtype=float)
    if trim_type not in ("value", "quantile", "discard"):
        raise ValueError("trim_type must be 'value', 'quantile' or 'discard'")
    if trim is not None and trim_type != "discard":
        lo, hi = float(trim[0]), float(trim[1])
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError("trim must satisfy 0 <= lo < hi <= 1")
        if trim_type == "quantile":
            v = sorted(float(u) for u in ps)
            lo = _quantile7(v, lo)
            hi = _quantile7(v, hi)
        ps = np.asarray([min(max(float(u), lo), hi) for u in ps], dtype=float)
    # numerical guard: the weights must stay finite whatever was asked
    return np.asarray([min(max(float(u), _PS_EPS), 1.0 - _PS_EPS)
                       for u in ps], dtype=float)


def _quantile7(sorted_v, p):
    """Type-7 sample quantile, matching R's stats::quantile default."""
    n = len(sorted_v)
    if n == 0:
        raise ValueError("empty propensity vector")
    if n == 1:
        return sorted_v[0]
    h = (n - 1) * p
    j = int(h)
    if j >= n - 1:
        return sorted_v[n - 1]
    g = h - j
    return sorted_v[j] * (1.0 - g) + sorted_v[j + 1] * g


def _trim_weights(w, weight_trim=None, side="upper"):
    """Cap IPW WEIGHTS at percentile cutpoints -- Lee, Lessler and
    Stuart (2011), PLoS ONE 6(3) e18174, who cap the high side only,
    hence side="upper" by default.  Contrast _trim_ps, which caps the
    SCORES.  Mirrored by .mor_trim_weights in the R arm."""
    if weight_trim is None:
        return w
    if side not in ("upper", "both"):
        raise ValueError("weight_trim_side must be 'upper' or 'both'")
    q = [float(v) for v in (weight_trim if hasattr(weight_trim, "__len__")
                            else (0.0, weight_trim))]
    if len(q) == 1:
        q = [0.0, q[0]]
    lo, hi = q[0], q[1]
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("weight_trim must satisfy 0 <= lo < hi <= 1")
    v = sorted(float(u) for u in w)
    clo, chi = _quantile7(v, lo), _quantile7(v, hi)
    if side == "both":
        return np.asarray([min(max(float(u), clo), chi) for u in w], dtype=float)
    return np.asarray([min(float(u), chi) for u in w], dtype=float)


def _om_ols(X, y):
    """Least squares by the normal equations; shared with the R arm."""
    p = len(X[0])
    A = [[sum(X[i][a] * X[i][b] for i in range(len(X))) for b in range(p)]
         for a in range(p)]
    rhs = [sum(X[i][a] * y[i] for i in range(len(X))) for a in range(p)]
    return _ps_solve(A, rhs)


def _om_fit_predict(X, y, rows, Xpred, outcome_model):
    """Fit the outcome model on `rows` and predict for every row of
    Xpred.  Linear -> OLS, logistic -> unpenalised IRLS."""
    Xs = [X[i] for i in rows]
    ys = [y[i] for i in rows]
    if outcome_model == "logistic":
        beta = _ps_irls_beta(Xs, ys, lam=0.0)
        out = []
        for row in Xpred:
            e = sum(row[j] * beta[j] for j in range(len(beta)))
            e = max(-30.0, min(30.0, e))
            out.append(1.0 / (1.0 + _math.exp(-e)))
        return out
    beta = _om_ols(Xs, ys)
    return [sum(row[j] * beta[j] for j in range(len(beta))) for row in Xpred]


def estimate_aipw(
    data: pd.DataFrame,
    *,
    treatment: str = "cannabis_any_use",
    outcome: str = "heavy_drinking_30d",
    covariates: list[str] | None = None,
    outcome_model: str = "logistic",
    trim: tuple[float, float] | None = (0.01, 0.99),
    trim_type: str = "value",
    ps_model: str = "mle",
    ridge_lambda: float = 1.0,
    outcome_fit: str = "separate",
) -> dict[str, Any]:
    """Augmented inverse-probability-weighted ATE.

    Propensity trimming
    -------------------
    Extreme propensity scores make the inverse-probability weights
    explode, so some safeguard is always applied.  BOTH routes in use
    are available here and the choice is explicit:

    ``trim_type="value"``
        Clamp the scores to the absolute bounds ``trim``.  Sample
        independent, so a stratified fit cannot be destabilised by a
        small stratum's own quantiles.  This is the default.
    ``trim_type="quantile"``
        Winsorise the scores at their own sample quantiles ``trim``.
        Percentile capping is the standard trimming device here: Lee,
        B. K., Lessler, J. and Stuart, E. A. (2011), "Weight trimming
        and propensity score weighting", PLoS ONE 6(3), e18174,
        doi:10.1371/journal.pone.0018174 -- "all weights with value
        above the [cutpoint] were set equal to the [cutpoint]".  See
        also Cole, S. R. and Hernan, M. A. (2008), Am J Epidemiol
        168(6), 656-664.
        Two honest differences from Lee et al.: they cap the WEIGHTS,
        this caps the SCORES (bounding the weights indirectly), and
        they cap the high side only while this caps both tails.
        It is NOT the rule of Crump, Hotz, Imbens and Mitnik (2009),
        Biometrika 96(1), 187-199: verified against that paper, they
        DISCARD units outside a range (rule of thumb [0.1, 0.9]),
        which changes the estimand.  Neither route here discards.

    ``trim=None``
        No trimming beyond the numerical guard that keeps the weights
        finite.

    The R arm ``morie_estimate_aipw`` takes the same two arguments with
    the same meanings and the same defaults.
    """
    r"""
    Estimate the ATE via the Augmented Inverse Probability Weighting (AIPW)
    doubly-robust estimator.

    The influence-function score for unit *i* is:

    .. math::

        \\psi_i =
            \\hat{\\mu}_1(X_i) - \\hat{\\mu}_0(X_i)
            + \\frac{T_i \\bigl(Y_i - \\hat{\\mu}_1(X_i)\\bigr)}{\\hat{e}_i}
            - \\frac{(1-T_i)\\bigl(Y_i - \\hat{\\mu}_0(X_i)\\bigr)}{1 - \\hat{e}_i}

    The ATE estimator is :math:`\\widehat{\\text{ATE}}_\\text{AIPW} = n^{-1} \\sum_i \\psi_i`
    and the standard error is :math:`\\hat{\\sigma}_{\\psi} / \\sqrt{n}`.

    **Double robustness**: the estimator is consistent if *either* the
    propensity score model :math:`\\hat{e}(X)` or the outcome model
    :math:`\\hat{\\mu}(T, X)` is correctly specified -- not necessarily both.

    :param data: The input DataFrame.
    :type data: pandas.DataFrame
    :param treatment: Binary treatment column (0/1).
    :type treatment: str
    :param outcome: Outcome column.
    :type outcome: str
    :param covariates: Covariate column names.  Defaults to the standard
        CPADS confounders.
    :type covariates: list[str] | None
    :param outcome_model: ``"logistic"`` for binary outcomes, ``"linear"``
        for continuous.  Defaults to ``"logistic"``.
    :type outcome_model: str
    :return: Dictionary with keys ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``n``, ``method``.
    :rtype: dict[str, Any]

    References
    ----------
    Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of
    regression coefficients when some regressors are not always observed.
    *Journal of the American Statistical Association*, 89(427), 846--866.

    Scharfstein, D. O., Rotnitzky, A., & Robins, J. M. (1999). Adjusting for
    nonignorable drop-out using semiparametric nonresponse models. *JASA*,
    94(448), 1096--1120.
    """
    covariates = covariates or [
        "age_group",
        "gender",
        "province_region",
        "mental_health",
        "physical_health",
    ]
    required = [treatment, outcome, *covariates]
    frame = data.loc[:, required].dropna().copy()

    t = frame[treatment].values.astype(float)
    y = frame[outcome].values.astype(float)

    # -- Propensity scores -------------------------------------------------------
    ps = compute_propensity_scores(frame, treatment=treatment,
                                   covariates=covariates,
                                   ps_model=ps_model,
                                   ridge_lambda=ridge_lambda).values
    ps = _trim_ps(ps, trim, trim_type)

    # -- Outcome model -----------------------------------------------------------
    # Two routes, both available and both matched exactly by the R arm:
    #   "separate" (default) fits E[Y | X, T = t] on each arm, so the
    #              covariate slopes may differ between treated and
    #              control -- the usual AIPW form;
    #   "pooled"   fits one regression on Y ~ T + X and predicts with T
    #              set to 1 and to 0, imposing a common slope.
    # Before 2026-08-12 the R arm was pooled and this one separate,
    # silently, which is why their estimates disagreed.
    if outcome_fit not in ("separate", "pooled"):
        raise ValueError("outcome_fit must be 'separate' or 'pooled'")
    Xc = _ps_design(frame, covariates)
    # Crump et al. discard route: drop units outside the overlap range
    # before fitting anything, and remember how many went.
    keep = _ps_keep(ps, trim, trim_type)
    n_discarded = int(sum(1 for k in keep if not k))
    if n_discarded:
        idx = [i for i, k in enumerate(keep) if k]
        if len(idx) < 2:
            raise ValueError("discard trimming removed almost every unit")
        t = np.asarray([t[i] for i in idx], dtype=float)
        y = np.asarray([y[i] for i in idx], dtype=float)
        ps = np.asarray([ps[i] for i in idx], dtype=float)
        Xc = [Xc[i] for i in idx]
    n = len(Xc)
    if outcome_fit == "pooled":
        Xp = [[Xc[i][0], t[i]] + Xc[i][1:] for i in range(n)]
        X1 = [[Xc[i][0], 1.0] + Xc[i][1:] for i in range(n)]
        X0 = [[Xc[i][0], 0.0] + Xc[i][1:] for i in range(n)]
        rows = list(range(n))
        mu1 = _om_fit_predict(Xp, y, rows, X1, outcome_model)
        mu0 = _om_fit_predict(Xp, y, rows, X0, outcome_model)
    else:
        r1 = [i for i in range(n) if t[i] == 1.0]
        r0 = [i for i in range(n) if t[i] == 0.0]
        mu1 = _om_fit_predict(Xc, y, r1, Xc, outcome_model)
        mu0 = _om_fit_predict(Xc, y, r0, Xc, outcome_model)
    mu1 = np.asarray(mu1, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)

    # -- AIPW influence scores ---------------------------------------------------
    psi = mu1 - mu0 + t * (y - mu1) / ps - (1.0 - t) * (y - mu0) / (1.0 - ps)

    n = len(psi)
    ate = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(n))
    z = 1.959964  # 97.5th percentile of standard normal

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "n": n,
        "method": "AIPW (doubly robust)",
    }


aipw = estimate_aipw


def cheatsheet() -> str:
    return "estimate_aipw({}) -> Augmented Inverse Probability Weighting (AIPW) doubly-robust"


# compact alias per ledger/NAMING.md
estimateaipw = estimate_aipw
