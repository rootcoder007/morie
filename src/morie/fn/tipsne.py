"""Tipping-point (delta-adjusted) sensitivity analysis for missingness.

A trial analysed under MAR gets one number. The question a tipping-point
analysis answers is a different one: how much worse would the unseen
outcomes of the people who dropped out have to be before the conclusion
stops holding? The answer is a quantity -- the delta at which the result
tips -- and it is reported so that a clinician can decide whether a
departure that large is plausible, which is a judgement no model makes.

The procedure, which is the one regulators ask for:

  1. Impute the missing outcomes under MAR, m times.
  2. Add a shift delta to the imputed values -- AFTER imputation and
     BEFORE analysis, which is the whole point: the imputation model
     stays MAR and the MNAR departure is applied on top of it, so the
     size of the departure is explicit rather than buried in a model.
     The shift is applied per arm, so the grid is two-dimensional.
  3. Analyse each completed data set (ANCOVA: outcome on arm plus any
     covariates) and pool with Rubin's rules.
  4. Sweep delta and find where the p-value crosses alpha. That
     crossing is the tipping point, located by linear interpolation
     between the two grid points that bracket it.

Routes, all selectable

  mi = "proper"     Rubin's proper multiple imputation: for each of the
                    m data sets, draw sigma^2 from its scaled inverse
                    chi-square posterior and beta from N(betahat,
                    sigma^2 (X'X)^-1), then draw the missing values
                    from the resulting predictive distribution. The
                    parameter draw is what makes the between-imputation
                    variance an honest estimate of the uncertainty in
                    the imputation model rather than only of the
                    residual noise.
  mi = "improper"   Impute from the fitted mean plus residual noise at
                    the point estimates, no parameter draw. Understates
                    the variance; included because it is what a
                    hand-rolled imputation usually does, and having it
                    side by side makes the size of that understatement
                    visible.
  mi = "deterministic"
                    One imputation at the fitted mean, no noise at all.
                    Not multiple imputation and not defensible as
                    inference, but it is the only route whose answer
                    does not move with the seed, so it is what the
                    anchors use to pin the arithmetic.

  pooling = "rubin1987"      df = (m-1) (1 + Ubar / ((1 + 1/m) B))^2
  pooling = "barnard_rubin"  the small-sample correction, which matters
                             here because a trial has a finite complete-
                             data df and Rubin's original formula can
                             hand back a df larger than the complete-data
                             one, which is nonsense.

Everything is written out in exact arithmetic -- compensated sums, an
explicit Cholesky, explicit triangular solves -- because the R arm has
to reproduce it bit for bit and `lm()` and `%*%` do not promise that.

References
  Rubin, D.B. (1987) "Multiple Imputation for Nonresponse in Surveys."
    Wiley. Chapter 3: the combining rules.
  Barnard, J. and Rubin, D.B. (1999) "Small-sample degrees of freedom
    with multiple imputation." Biometrika 86(4), 948-955.
  Yan, X., Lee, S., Ling, N. and Lin, J. (2021) tipping-point
    sensitivity analysis for MNAR departures in clinical trials; the
    delta-adjustment procedure as implemented here follows the
    now-standard regulatory form (impute under MAR, shift imputed
    values by delta per arm, re-analyse, locate the crossing), as
    documented for the `rbmi` and SAS implementations.
"""

import math

from . import _array_core as _core
from ._richresult import RichResult

__all__ = ["tipping_point_sensitivity", "tipsne", "ancova_fit",
           "rubin_pool", "impute_once", "cheatsheet", "MI_ROUTES",
           "POOLING_ROUTES"]

MI_ROUTES = ("proper", "improper", "deterministic")
POOLING_ROUTES = ("rubin1987", "barnard_rubin")


def _csum(vals):
    """Neumaier-compensated sum.

    Written out rather than left to sum(): CPython 3.12+ compensates a
    run of floats and R's sum() accumulates in long double, so the two
    built-ins are different functions and a comparison at the twelfth
    digit will find the difference.
    """
    s = 0.0
    c = 0.0
    for v in vals:
        t = float(v)
        u = s + t
        if abs(s) >= abs(t):
            c += (s - u) + t
        else:
            c += (t - u) + s
        s = u
    return s + c


def _dot(a, b):
    """Compensated dot product. Not `sum(x*y for ...)`, same reason."""
    s = 0.0
    c = 0.0
    for x, y in zip(a, b):
        t = float(x) * float(y)
        u = s + t
        if abs(s) >= abs(t):
            c += (s - u) + t
        else:
            c += (t - u) + s
        s = u
    return s + c


def _chol(a):
    """Cholesky factor L with A = L L', lower triangular.

    Explicit rather than a library call so the R arm can match it
    element by element.
    """
    p = len(a)
    lo = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i + 1):
            s = a[i][j] - _dot(lo[i][:j], lo[j][:j])
            if i == j:
                if s <= 0.0:
                    raise ValueError("design matrix is not full rank")
                lo[i][j] = math.sqrt(s)
            else:
                lo[i][j] = s / lo[j][j]
    return lo


def _solve_chol(lo, b):
    """Solve L L' x = b by forward then back substitution."""
    p = len(lo)
    z = [0.0] * p
    for i in range(p):
        z[i] = (b[i] - _dot(lo[i][:i], z[:i])) / lo[i][i]
    x = [0.0] * p
    for i in range(p - 1, -1, -1):
        acc = _csum(lo[k][i] * x[k] for k in range(i + 1, p))
        x[i] = (z[i] - acc) / lo[i][i]
    return x


def _inv_from_chol(lo):
    """(L L')^-1, formed column by column from the factor."""
    p = len(lo)
    cols = []
    for j in range(p):
        e = [1.0 if k == j else 0.0 for k in range(p)]
        cols.append(_solve_chol(lo, e))
    return [[cols[j][i] for j in range(p)] for i in range(p)]


def ancova_fit(y, design):
    """Least squares of y on `design`, via the normal equations.

    Parameters
    ----------
    y : list of float
    design : list of list of float
        One row per observation, including the intercept column.

    Returns
    -------
    dict
        beta, residual sum of squares, df, sigma2 and the unscaled
        covariance (X'X)^-1.
    """
    n = len(y)
    p = len(design[0])
    xtx = [[_csum(design[i][a] * design[i][b] for i in range(n))
            for b in range(p)] for a in range(p)]
    xty = [_csum(design[i][a] * y[i] for i in range(n)) for a in range(p)]
    lo = _chol(xtx)
    beta = _solve_chol(lo, xty)
    fitted = [_dot(design[i], beta) for i in range(n)]
    rss = _csum((y[i] - fitted[i]) * (y[i] - fitted[i]) for i in range(n))
    df = n - p
    if df < 1:
        raise ValueError("no residual degrees of freedom")
    return {"beta": beta, "rss": rss, "df": df, "sigma2": rss / df,
            "xtx_inv": _inv_from_chol(lo), "fitted": fitted, "chol": lo}


def _design(arm, X, n):
    """Intercept, arm indicator, then any covariates."""
    rows = []
    for i in range(n):
        row = [1.0, float(arm[i])]
        if X is not None:
            row.extend(float(v) for v in X[i])
        rows.append(row)
    return rows


def _draw_beta(rng, beta, xtx_inv, sigma2_draw):
    """beta* ~ N(betahat, sigma2 (X'X)^-1), via the Cholesky of the
    covariance. The draw is coordinate by coordinate so the stream
    position matches the R arm term for term."""
    p = len(beta)
    cov = [[sigma2_draw * xtx_inv[i][j] for j in range(p)]
           for i in range(p)]
    lo = _chol(cov)
    z = [float(rng.normal()) for _ in range(p)]
    return [beta[i] + _dot(lo[i][:i + 1], z[:i + 1]) for i in range(p)]


def impute_once(rng, y, arm, X, miss, fit, mi):
    """One completed outcome vector under the chosen imputation route."""
    n = len(y)
    des = _design(arm, X, n)
    if mi == "deterministic":
        beta, sd = fit["beta"], 0.0
    elif mi == "improper":
        beta, sd = fit["beta"], math.sqrt(fit["sigma2"])
    else:
        # sigma2* = rss / chi2_df, the scaled inverse chi-square
        # posterior draw; chi2 is Gamma(df/2, 2), which is the shape the
        # matched generator provides in both arms.
        g = float(rng.gamma(fit["df"] / 2.0, 2.0))
        sigma2 = fit["rss"] / g
        beta = _draw_beta(rng, fit["beta"], fit["xtx_inv"], sigma2)
        sd = math.sqrt(sigma2)
    out = list(y)
    for i in range(n):
        if miss[i]:
            mu = _dot(des[i], beta)
            out[i] = mu + (sd * float(rng.normal()) if sd > 0.0 else 0.0)
    return out


def _t_sf(t, df):
    """Upper tail of Student's t, from the regularised incomplete beta.

    Written out because the two languages' distribution functions are
    separate implementations and would disagree in the last digits.
    """
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2.0, 0.5, x)


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b) by the continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (_lgamma(a) + _lgamma(b) - _lgamma(a + b))
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(b * math.log(1.0 - x) + a * math.log(x)
                          - lbeta) * _betacf(b, a, 1.0 - x) / b


def _betacf(a, b, x):
    """Lentz's algorithm for the beta continued fraction."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return h


_LG = (76.18009172947146, -86.50532032941677, 24.01409824083091,
       -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5)


def _lgamma(z):
    """Lanczos log-gamma, written out so both arms use the same one."""
    x = z
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for j in range(6):
        x += 1.0
        ser += _LG[j] / x
    return -tmp + math.log(2.5066282746310005 * ser / z)


def rubin_pool(ests, vars_, pooling="rubin1987", df_complete=None):
    """Combine per-imputation estimates and variances.

    Parameters
    ----------
    ests, vars_ : list of float
        Point estimates and their squared standard errors, one per
        imputation.
    pooling : str
        "rubin1987" or "barnard_rubin".
    df_complete : float or None
        Complete-data residual df; required by the Barnard-Rubin route.

    Returns
    -------
    dict
        estimate, se, df, t, p and the variance decomposition.
    """
    m = len(ests)
    qbar = _csum(ests) / m
    ubar = _csum(vars_) / m
    if m > 1:
        b = _csum((e - qbar) * (e - qbar) for e in ests) / (m - 1)
    else:
        b = 0.0
    total = ubar + (1.0 + 1.0 / m) * b
    if b <= 0.0 or m < 2:
        # No between-imputation variance: the imputation added nothing,
        # so the complete-data df is the honest answer and Rubin's
        # formula would divide by zero.
        df = df_complete if df_complete is not None else 1e6
    else:
        r = (1.0 + 1.0 / m) * b / ubar
        df = (m - 1) * (1.0 + 1.0 / r) * (1.0 + 1.0 / r)
        if pooling == "barnard_rubin":
            if df_complete is None:
                raise ValueError("barnard_rubin needs df_complete")
            gamma = (1.0 + 1.0 / m) * b / total
            dfo = ((df_complete + 1.0) / (df_complete + 3.0)
                   * df_complete * (1.0 - gamma))
            df = 1.0 / (1.0 / df + 1.0 / dfo)
    se = math.sqrt(total)
    t = qbar / se if se > 0.0 else 0.0
    return {"estimate": qbar, "se": se, "df": df, "t": t,
            "p": 2.0 * _t_sf(abs(t), df), "within": ubar, "between": b,
            "total": total,
            "fmi": ((1.0 + 1.0 / m) * b / total) if total > 0.0 else 0.0}


def _sd(vals):
    n = len(vals)
    mu = _csum(vals) / n
    return math.sqrt(_csum((v - mu) * (v - mu) for v in vals) / (n - 1))


def tipping_point_sensitivity(y, D, missing_indicator=None, X=None,
                              delta_treat=None, delta_control=None,
                              n_imputations=20, seed=1, alpha=0.05,
                              mi="proper", pooling="rubin1987",
                              standardise=True):
    """Delta-adjusted tipping-point sensitivity analysis.

    Parameters
    ----------
    y : sequence
        Outcome. Entries that are missing may be given as None or NaN.
    D : sequence
        Arm indicator, 0 for control and 1 for treatment.
    missing_indicator : sequence or None
        1 where the outcome is missing. Derived from `y` when omitted;
        when both are given they must agree, and a disagreement raises
        rather than silently picking one.
    X : sequence of sequences or None
        Covariates for the ANCOVA, complete for every unit.
    delta_treat, delta_control : sequence or None
        Shifts applied to imputed values in each arm. The defaults
        sweep the treated arm from 0 down to -2.5 pooled standard
        deviations in eleven steps, with the control arm held at 0,
        which is the one-way analysis; pass both to get a two-way grid.
    n_imputations : int
        m. Ignored by the "deterministic" route, which uses one.
    seed : int
        Seed for the shared generator. Every grid cell is imputed from
        the SAME seed, so a difference between two cells is the delta
        and not the draws -- otherwise the sweep would be noise.
    alpha : float
        Significance level the crossing is measured against.
    mi : str
        "proper", "improper" or "deterministic".
    pooling : str
        "rubin1987" or "barnard_rubin".
    standardise : bool
        Report the deltas in pooled standard deviations as well as in
        the outcome's own units.

    Returns
    -------
    RichResult
        The MAR analysis, the full grid, the tipping point for each
        control-arm delta, and whether the result tipped inside the
        grid at all.

    References
    ----------
    Rubin (1987) ch. 3; Barnard and Rubin (1999) Biometrika 86, 948-955.
    """
    if mi not in MI_ROUTES:
        raise ValueError("mi must be one of %r" % (MI_ROUTES,))
    if pooling not in POOLING_ROUTES:
        raise ValueError("pooling must be one of %r" % (POOLING_ROUTES,))
    yv = [None if v is None or v != v else float(v) for v in y]
    n = len(yv)
    arm = [float(v) for v in D]
    derived = [1 if v is None else 0 for v in yv]
    if missing_indicator is None:
        miss = derived
    else:
        miss = [1 if v else 0 for v in missing_indicator]
        if miss != derived and any(miss[i] == 0 and derived[i] == 1
                                   for i in range(n)):
            raise ValueError("missing_indicator says observed where y "
                             "is missing")
    if X is not None:
        X = [[float(v) for v in row] for row in X]
    obs = [i for i in range(n) if not miss[i]]
    if len(obs) < 3:
        raise ValueError("fewer than three observed outcomes")

    des_all = _design(arm, X, n)
    fit = ancova_fit([yv[i] for i in obs], [des_all[i] for i in obs])
    pooled_sd = _sd([yv[i] for i in obs])
    df_complete = n - len(des_all[0])

    if delta_treat is None:
        step = 2.5 * pooled_sd / 10.0
        delta_treat = [-step * k for k in range(11)]
    else:
        delta_treat = [float(v) for v in delta_treat]
    if delta_control is None:
        delta_control = [0.0]
    else:
        delta_control = [float(v) for v in delta_control]

    m = 1 if mi == "deterministic" else int(n_imputations)

    def cell(dc, dt):
        rng = _core._SplitMix64(seed)
        ests = []
        vars_ = []
        for _ in range(m):
            comp = impute_once(rng, yv, arm, X, miss, fit, mi)
            for i in range(n):
                if miss[i]:
                    comp[i] = comp[i] + (dt if arm[i] == 1.0 else dc)
            f = ancova_fit(comp, des_all)
            ests.append(f["beta"][1])
            vars_.append(f["sigma2"] * f["xtx_inv"][1][1])
        return rubin_pool(ests, vars_, pooling, df_complete)

    mar = cell(0.0, 0.0)

    grid = []
    tips = []
    for dc in delta_control:
        row = [cell(dc, dt) for dt in delta_treat]
        for dt, r in zip(delta_treat, row):
            grid.append({"delta_control": dc, "delta_treat": dt,
                         "estimate": r["estimate"], "se": r["se"],
                         "df": r["df"], "p": r["p"],
                         "significant": r["p"] < alpha})
        # The crossing, by linear interpolation between the two grid
        # points that bracket it. Reported as None when the row never
        # crosses -- an extrapolated tipping point outside the grid
        # would be a number the data does not support.
        tp = None
        for k in range(1, len(delta_treat)):
            p0, p1 = row[k - 1]["p"], row[k]["p"]
            if (p0 < alpha) != (p1 < alpha) and p1 != p0:
                w = (alpha - p0) / (p1 - p0)
                tp = delta_treat[k - 1] + w * (delta_treat[k]
                                               - delta_treat[k - 1])
                break
        tips.append({"delta_control": dc, "tipping_point": tp,
                     "tipping_point_sd": (None if tp is None or
                                          pooled_sd == 0.0
                                          else tp / pooled_sd)})

    payload = {
        "estimate": mar["estimate"],
        "se": mar["se"],
        "p": mar["p"],
        "df": mar["df"],
        "mar": mar,
        "grid": grid,
        "tipping_points": tips,
        "tipped": any(t["tipping_point"] is not None for t in tips),
        "n": n,
        "n_missing": sum(miss),
        "n_missing_treat": sum(1 for i in range(n)
                               if miss[i] and arm[i] == 1.0),
        "n_missing_control": sum(1 for i in range(n)
                                 if miss[i] and arm[i] == 0.0),
        "pooled_sd": pooled_sd if standardise else None,
        "m": m,
        "mi": mi,
        "pooling": pooling,
        "alpha": float(alpha),
        "seed": int(seed),
        "method": "delta-adjusted tipping-point sensitivity analysis",
    }
    return RichResult(payload=payload)


tipsne = tipping_point_sensitivity


def cheatsheet():
    return ("tipsne: delta-adjusted tipping-point sensitivity analysis "
            "for MNAR missingness. mi routes " + ", ".join(MI_ROUTES) +
            "; pooling " + ", ".join(POOLING_ROUTES))
