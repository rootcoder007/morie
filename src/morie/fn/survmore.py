"""The fifteen survival methods that de-externalization dropped.

Commit 508cdd7 replaced R/survival.R with native Kaplan-Meier,
Nelson-Aalen, log-rank, Cox and concordance, correctly removing the
survival:: and cmprsk:: wrappers.  Five of the old entry points were
renamed onto those natives; these fifteen had no replacement and were
simply deleted, leaving NAMESPACE exporting names that did not exist.

Everything here is written from the definitions, with no external
statistical package.  Where the old wrapper delegated to survival:: or
cmprsk::, the docstring says which estimator is being reproduced and
where the two would differ.
"""

from math import exp, fsum, inf, isfinite, log, sqrt

from . import _sci_core as _sc
from ._richresult import RichResult

__all__ = [
    "rmst", "rmstdiff", "martingale", "devresid", "coxsnell",
    "schoenfeld", "hazratio", "cif", "finegray", "ltkm", "landmark",
    "turnbull", "parasurv", "aftfit", "paracompare",
]


# ------------------------------------------------------------------ utils

def _flat(v):
    if hasattr(v, "tolist"):
        v = v.tolist()
    return [float(x) for x in v]


def _ints(v):
    if hasattr(v, "tolist"):
        v = v.tolist()
    return [int(x) for x in v]


def _mat(X):
    if hasattr(X, "tolist"):
        X = X.tolist()
    rows = [list(r) if hasattr(r, "__len__") else [r] for r in X]
    if not rows:
        raise ValueError("the design matrix is empty")
    p = len(rows[0])
    if any(len(r) != p for r in rows):
        raise ValueError("every row of X must have the same length")
    return [[float(v) for v in r] for r in rows], p


def _check(time, event):
    t, e = _flat(time), _ints(event)
    if len(t) != len(e):
        raise ValueError("time and event must have the same length")
    if not t:
        raise ValueError("need at least one observation")
    if any(v < 0 for v in t):
        raise ValueError("follow-up times cannot be negative")
    if any(v not in (0, 1) for v in e):
        raise ValueError("event must be 0 (censored) or 1 (event)")
    return t, e


def _norm_q(p):
    """Standard normal quantile, Wichura's AS 241 via the native core."""
    from ._stats_core import _norm_ppf

    return _norm_ppf(p)


def _norm_sf(x):
    from ._sci_core import erfc

    return 0.5 * erfc(x / sqrt(2.0))


def _chi2_sf(x, df):
    from ._sci_core import gammaincc

    return gammaincc(df / 2.0, x / 2.0)


def _km_curve(time, event):
    """Kaplan-Meier steps: (event times, S, n_risk, n_event, greenwood)."""
    t, e = _check(time, event)
    ut = sorted(set(t[i] for i in range(len(t)) if e[i] == 1))
    S, nr, ne, gw = [], [], [], []
    surv, v = 1.0, 0.0
    for u in ut:
        n_i = sum(1 for x in t if x >= u)
        d_i = sum(1 for i in range(len(t)) if t[i] == u and e[i] == 1)
        surv *= (1.0 - d_i / n_i)
        if n_i > d_i:
            v += d_i / (n_i * (n_i - d_i))
        S.append(surv)
        nr.append(n_i)
        ne.append(d_i)
        gw.append(v)
    return ut, S, nr, ne, gw


def _breslow_baseline(time, event, X, beta):
    """Breslow cumulative baseline hazard at each event time.

    H0(t) = sum_{t_i <= t} d_i / sum_{j in R(t_i)} exp(x_j' beta),
    the estimator survival::basehaz returns for a coxph fit.
    """
    t, e = _check(time, event)
    Xm, p = _mat(X)
    if len(Xm) != len(t):
        raise ValueError("X must have one row per observation")
    if len(beta) != p:
        raise ValueError("beta must have one entry per column of X")
    w = [exp(fsum(Xm[i][k] * beta[k] for k in range(p)))
         for i in range(len(t))]
    ut = sorted(set(t[i] for i in range(len(t)) if e[i] == 1))
    H, cum = [], 0.0
    for u in ut:
        d_i = sum(1 for i in range(len(t)) if t[i] == u and e[i] == 1)
        denom = fsum(w[i] for i in range(len(t)) if t[i] >= u)
        if denom <= 0:
            raise ValueError("empty risk set at t = %g" % u)
        cum += d_i / denom
        H.append(cum)
    return ut, H, w


def _h0_at(ut, H, t):
    """Step-function lookup: H0 at the last event time not exceeding t."""
    out, j = [], 0
    for x in t:
        while j < len(ut) and ut[j] <= x:
            j += 1
        out.append(H[j - 1] if j else 0.0)
        j = 0
    # the loop above resets j each time; do it properly in one pass
    out = []
    for x in t:
        lo, hi, best = 0, len(ut) - 1, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if ut[mid] <= x:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        out.append(H[best] if best >= 0 else 0.0)
    return out


# -------------------------------------------------- restricted mean, 1 & 2

def rmst(time, event, tau=None, alpha=0.05):
    """Restricted mean survival time.

    RMST(tau) = integral_0^tau S(t) dt, the area under the Kaplan-Meier
    curve out to a horizon tau, with the variance of Klein and
    Moeschberger (2003) eq. (4.5.4):

        Var = sum_{t_i <= tau} [ integral_{t_i}^{tau} S(u) du ]^2
                               d_i / ( n_i (n_i - d_i) ).

    The horizon is not optional in substance.  RMST is only estimable
    out to the largest observed time, and beyond the last EVENT the
    curve is flat at a value the data cannot pin down; a tau past the
    end of follow-up silently reports the plateau as if it were
    estimated.  With tau unset it defaults to the largest observed time
    and ``tau_beyond_data`` reports whether the supplied one overruns.

    RMST answers the question a hazard ratio does not: how much longer,
    on average, over a fixed window.  It needs no proportional-hazards
    assumption, which is why it is preferred when curves cross.
    """
    t, e = _check(time, event)
    ut, S, nr, ne, _ = _km_curve(t, e)
    tmax = max(t)
    horizon = float(tau) if tau is not None else tmax
    if horizon <= 0:
        raise ValueError("tau must be positive")
    # area of the step function up to the horizon
    area, prev_t, prev_S = 0.0, 0.0, 1.0
    steps = []
    for u, s in zip(ut, S):
        if u >= horizon:
            break
        area += prev_S * (u - prev_t)
        steps.append((u, prev_S))
        prev_t, prev_S = u, s
    area += prev_S * (horizon - prev_t)
    # variance: tail area from each event time to the horizon
    var = 0.0
    for i, u in enumerate(ut):
        if u > horizon:
            break
        if nr[i] <= ne[i]:
            continue
        tail, pt, ps = 0.0, u, S[i]
        for v, s in zip(ut[i + 1:], S[i + 1:]):
            if v >= horizon:
                break
            tail += ps * (v - pt)
            pt, ps = v, s
        tail += ps * (horizon - pt)
        var += tail * tail * ne[i] / (nr[i] * (nr[i] - ne[i]))
    se = sqrt(var) if var > 0 else 0.0
    z = _norm_q(1.0 - alpha / 2.0)
    return RichResult(payload={
        "rmst": area, "se": se, "variance": var, "tau": horizon,
        "lower": area - z * se, "upper": area + z * se,
        "max_time": tmax, "max_event_time": ut[-1] if ut else 0.0,
        "tau_beyond_data": horizon > tmax,
        "tau_beyond_last_event": bool(ut) and horizon > ut[-1],
        "n": len(t), "n_events": sum(e),
        "method": "Klein and Moeschberger (2003) eq. (4.5.4); area under "
                  "the Kaplan-Meier curve"})


def rmstdiff(time, event, group, tau=None, alpha=0.05):
    """Difference in restricted mean survival time between two groups.

        RMST_1(tau) - RMST_2(tau),  SE = sqrt(V_1 + V_2),

    the two groups being independent so the variances add.

    tau MUST be common to both arms, and it is capped at the smaller of
    the two groups' largest observed times: comparing areas over
    different windows compares different quantities, and the shorter
    arm cannot support the longer window.  ``tau_capped`` says when that
    happened.
    """
    t, e = _check(time, event)
    g = list(group)
    if len(g) != len(t):
        raise ValueError("group must have one entry per observation")
    levels = sorted(set(g), key=str)
    if len(levels) != 2:
        raise ValueError("rmstdiff compares exactly two groups, got %d"
                         % len(levels))
    parts = []
    for lv in levels:
        idx = [i for i in range(len(t)) if g[i] == lv]
        if not idx:
            raise ValueError("group %r is empty" % (lv,))
        parts.append(([t[i] for i in idx], [e[i] for i in idx]))
    cap = min(max(p[0]) for p in parts)
    horizon = min(float(tau), cap) if tau is not None else cap
    capped = tau is not None and float(tau) > cap
    a = rmst(parts[0][0], parts[0][1], tau=horizon, alpha=alpha)
    b = rmst(parts[1][0], parts[1][1], tau=horizon, alpha=alpha)
    diff = a["rmst"] - b["rmst"]
    se = sqrt(a["variance"] + b["variance"])
    z = _norm_q(1.0 - alpha / 2.0)
    stat = diff / se if se > 0 else 0.0
    return RichResult(payload={
        "difference": diff, "se": se, "z": stat,
        "p_value": 2.0 * _norm_sf(abs(stat)),
        "lower": diff - z * se, "upper": diff + z * se,
        "tau": horizon, "tau_capped": capped, "levels": levels,
        "rmst": [a["rmst"], b["rmst"]],
        "ratio": (a["rmst"] / b["rmst"]) if b["rmst"] else None,
        "method": "difference of restricted means over a COMMON horizon"})


# ------------------------------------------------------------- residuals

def martingale(time, event, X, beta):
    """Martingale residuals from a fitted Cox model.

        M_i = delta_i - H0(t_i) exp(x_i' beta),

    the observed number of events for subject i minus the number the
    model expects over its follow-up.  They sum to zero at the fitted
    beta, which is the check returned.

    Their use is functional form: plotting M_i from a NULL model against
    a candidate covariate shows the transformation of it the model
    wants.  They are badly skewed -- bounded above by 1 and unbounded
    below -- so they are poor for spotting outliers; the deviance
    residuals of :func:`devresid` are the symmetrized version for that.
    """
    t, e = _check(time, event)
    ut, H, w = _breslow_baseline(t, e, X, beta)
    h = _h0_at(ut, H, t)
    m = [e[i] - h[i] * w[i] for i in range(len(t))]
    return RichResult(payload={
        "residuals": m, "expected": [h[i] * w[i] for i in range(len(t))],
        "sum": fsum(m), "sums_to_zero": abs(fsum(m)) < 1e-6 * len(t),
        "max": max(m), "min": min(m), "upper_bound": 1.0,
        "n": len(t), "skewed": True,
        "method": "M_i = delta_i - H0(t_i) exp(x_i' beta), Breslow "
                  "baseline"})


def devresid(time, event, X, beta):
    """Deviance residuals from a fitted Cox model.

        d_i = sign(M_i) sqrt( -2 [ M_i + delta_i log(delta_i - M_i) ] ),

    a symmetrizing transform of the martingale residuals: roughly
    normal when the model fits, so a large |d_i| is an outlier in the
    usual sense.

    The log term is taken as zero when delta_i = 0, which is the limit
    of delta log(delta - M) as delta -> 0 and not a special case being
    swept aside.  Their sum of squares is NOT the model deviance for a
    Cox fit, since the partial likelihood is not a full likelihood; the
    name is by analogy.
    """
    t, e = _check(time, event)
    m = martingale(t, e, X, beta)["residuals"]
    d = []
    for i in range(len(t)):
        inner = m[i]
        if e[i]:
            arg = e[i] - m[i]
            if arg <= 0:
                raise ValueError("delta - M is not positive at i = %d; the "
                                 "deviance residual is undefined there" % i)
            inner = m[i] + e[i] * log(arg)
        val = -2.0 * inner
        s = 1.0 if m[i] >= 0 else -1.0
        d.append(s * sqrt(val if val > 0 else 0.0))
    return RichResult(payload={
        "residuals": d, "martingale": m,
        "sum_of_squares": fsum(v * v for v in d),
        "max_abs": max(abs(v) for v in d), "n": len(t),
        "is_model_deviance": False,
        "method": "d_i = sign(M) sqrt(-2[M + delta log(delta - M)])"})


def coxsnell(time, event, X, beta):
    """Cox-Snell residuals from a fitted Cox model.

        r_i = H0(t_i) exp(x_i' beta) = delta_i - M_i,

    the fitted cumulative hazard for each subject.  If the model is
    correct these behave like a censored sample from a unit
    exponential, so the Nelson-Aalen cumulative hazard OF THE RESIDUALS
    plotted against the residuals should follow the 45-degree line.

    That diagnostic curve is computed here and returned, because the
    residuals alone say nothing without it.  The check is weak in small
    samples -- the plot is fitted, not held out -- so a visually
    straight line is necessary and far from sufficient.
    """
    t, e = _check(time, event)
    ut, H, w = _breslow_baseline(t, e, X, beta)
    h = _h0_at(ut, H, t)
    r = [h[i] * w[i] for i in range(len(t))]
    rt, rH = [], []
    order = sorted(range(len(r)), key=lambda i: r[i])
    cum = 0.0
    for i in order:
        if e[i] != 1:
            continue
        n_i = sum(1 for j in range(len(r)) if r[j] >= r[i])
        d_i = sum(1 for j in range(len(r))
                  if r[j] == r[i] and e[j] == 1)
        cum += d_i / n_i
        rt.append(r[i])
        rH.append(cum)
    dev = max((abs(a - b) for a, b in zip(rt, rH)), default=0.0)
    return RichResult(payload={
        "residuals": r, "diagnostic_x": rt, "diagnostic_h": rH,
        "max_deviation": dev, "n": len(t),
        "reference": "unit exponential; the Nelson-Aalen hazard of the "
                     "residuals should follow the 45-degree line",
        "in_sample_check": True,
        "method": "r_i = H0(t_i) exp(x_i' beta) = delta_i - M_i"})


def schoenfeld(time, event, X, beta, vcov=None, scaled=True):
    """Schoenfeld residuals and the proportional-hazards test.

    At each event time, the residual is the covariate of the subject who
    failed minus the risk-set average weighted by the model:

        s_i = x_i - sum_{j in R} x_j w_j / sum_{j in R} w_j.

    Grambsch and Therneau (1994) scale them,

        s*_i = beta + d V s_i,

    with d the number of events and V the covariance of beta; under
    proportional hazards E[s*_i] = beta at every time, so a nonzero
    correlation between s* and (a transform of) time is evidence
    against PH.  That correlation test is returned.

    One event per residual: with ties, only the first event at a tied
    time gets a residual under this (Breslow-style) definition, and
    ``ties_dropped`` counts what that discarded.
    """
    t, e = _check(time, event)
    Xm, p = _mat(X)
    b = _flat(beta)
    if len(b) != p:
        raise ValueError("beta must have one entry per column of X")
    w = [exp(fsum(Xm[i][k] * b[k] for k in range(p)))
         for i in range(len(t))]
    ut = sorted(set(t[i] for i in range(len(t)) if e[i] == 1))
    times, res, dropped = [], [], 0
    for u in ut:
        rk = [i for i in range(len(t)) if t[i] >= u]
        ev = [i for i in range(len(t)) if t[i] == u and e[i] == 1]
        dropped += len(ev) - 1
        sw = fsum(w[i] for i in rk)
        xbar = [fsum(Xm[i][k] * w[i] for i in rk) / sw for k in range(p)]
        i0 = ev[0]
        times.append(u)
        res.append([Xm[i0][k] - xbar[k] for k in range(p)])
    out = {"time": times, "residuals": res, "n_events": len(res),
           "ties_dropped": dropped, "p": p,
           "method": "s_i = x_i - weighted risk-set mean"}
    if scaled:
        if vcov is None:
            raise ValueError("scaling needs the covariance of beta; pass "
                             "vcov= from the Cox fit, or scaled=False")
        V, q = _mat(vcov)
        if q != p or len(V) != p:
            raise ValueError("vcov must be p x p")
        d = float(len(res))
        sc = []
        for r in res:
            sc.append([b[k] + d * fsum(V[k][j] * r[j] for j in range(p))
                       for k in range(p)])
        out["scaled"] = sc
        # correlation of each scaled residual with time, and a z test
        stats = []
        for k in range(p):
            y = [row[k] for row in sc]
            n = len(y)
            if n < 3:
                stats.append({"rho": None, "z": None, "p_value": None})
                continue
            mt = fsum(times) / n
            my = fsum(y) / n
            st = fsum((v - mt) ** 2 for v in times)
            sy = fsum((v - my) ** 2 for v in y)
            if st <= 0 or sy <= 0:
                stats.append({"rho": 0.0, "z": 0.0, "p_value": 1.0})
                continue
            rho = fsum((a - mt) * (c - my)
                       for a, c in zip(times, y)) / sqrt(st * sy)
            z = rho * sqrt(n - 1)
            stats.append({"rho": rho, "z": z,
                          "p_value": 2.0 * _norm_sf(abs(z))})
        out["ph_test"] = stats
        out["ph_violated"] = [s["p_value"] is not None and s["p_value"] < 0.05
                              for s in stats]
        out["method"] = ("Grambsch and Therneau (1994) scaled Schoenfeld "
                         "residuals and the correlation-with-time PH test")
    return RichResult(payload=out)


def hazratio(beta, se, alpha=0.05, names=None):
    """Hazard ratios with confidence intervals from a Cox fit.

        HR = exp(beta),   CI = exp( beta +/- z se ).

    The interval is formed on the LOG scale and then exponentiated, so
    it is asymmetric about the HR and cannot cross zero.  Building it as
    HR +/- z se(HR) instead -- the common slip -- gives an interval that
    can include negative hazard ratios and has the wrong coverage.

    Interpretation is a ratio of instantaneous hazards, constant over
    time only if proportional hazards holds; check that with
    :func:`schoenfeld` before quoting a single number.
    """
    b, s = _flat(beta), _flat(se)
    if len(b) != len(s):
        raise ValueError("beta and se must have the same length")
    if not b:
        raise ValueError("need at least one coefficient")
    if any(v < 0 for v in s):
        raise ValueError("standard errors cannot be negative")
    z = _norm_q(1.0 - alpha / 2.0)
    hr = [exp(v) for v in b]
    lo = [exp(v - z * u) for v, u in zip(b, s)]
    hi = [exp(v + z * u) for v, u in zip(b, s)]
    zs = [(v / u if u > 0 else 0.0) for v, u in zip(b, s)]
    return RichResult(payload={
        "hazard_ratio": hr, "lower": lo, "upper": hi, "coef": b, "se": s,
        "z": zs, "p_value": [2.0 * _norm_sf(abs(v)) for v in zs],
        "names": list(names) if names is not None else None,
        "alpha": alpha, "interval_on_log_scale": True,
        "assumes_proportional_hazards": True,
        "method": "HR = exp(beta), interval exponentiated from the log "
                  "scale"})


# ---------------------------------------------------------- competing risks

def cif(time, cause, code=1, alpha=0.05):
    """Cumulative incidence function under competing risks.

    The Aalen-Johansen estimator:

        CIF_k(t) = sum_{t_i <= t} S(t_{i-1}) d_{ki} / n_i,

    where S is the ALL-CAUSE Kaplan-Meier and d_ki the events of cause k.

    The factor S(t_{i-1}) is the entire point.  Treating the competing
    events as censored and running an ordinary Kaplan-Meier gives
    1 - S_k, which OVERSTATES the incidence -- sometimes grossly -- by
    assuming the subjects who died of something else would otherwise
    have remained at risk of cause k.  Both are returned so the size of
    that bias is visible on the data at hand.

    Parameters
    ----------
    time : array-like
        Follow-up times.
    cause : array-like
        0 for censored, otherwise an integer cause label.
    code : int
        Which cause to estimate.
    """
    t = _flat(time)
    c = _ints(cause)
    if len(t) != len(c):
        raise ValueError("time and cause must have the same length")
    if not t:
        raise ValueError("need at least one observation")
    k = int(code)
    if k == 0:
        raise ValueError("cause 0 marks censoring; pick an event cause")
    if k not in c:
        raise ValueError("cause %d does not occur in the data" % k)
    ut = sorted(set(t[i] for i in range(len(t)) if c[i] != 0))
    surv = 1.0
    F, times, nr, nk, var = [], [], [], [], []
    cum, v = 0.0, 0.0
    for u in ut:
        n_i = sum(1 for x in t if x >= u)
        d_all = sum(1 for i in range(len(t)) if t[i] == u and c[i] != 0)
        d_k = sum(1 for i in range(len(t)) if t[i] == u and c[i] == k)
        cum += surv * d_k / n_i
        # Aalen's variance, the delta-method form
        if n_i > d_all:
            v += (surv ** 2) * d_k * (n_i - d_k) / (n_i ** 3)
        surv *= (1.0 - d_all / n_i)
        times.append(u)
        F.append(cum)
        nr.append(n_i)
        nk.append(d_k)
        var.append(v)
    # the naive "censor the competing events" curve, for contrast
    naive_e = [1 if c[i] == k else 0 for i in range(len(t))]
    _, S_naive, _, _, _ = _km_curve(t, naive_e)
    naive = [1.0 - s for s in S_naive]
    z = _norm_q(1.0 - alpha / 2.0)
    se = [sqrt(x) if x > 0 else 0.0 for x in var]
    return RichResult(payload={
        "time": times, "cif": F, "se": se,
        "lower": [max(0.0, a - z * b) for a, b in zip(F, se)],
        "upper": [min(1.0, a + z * b) for a, b in zip(F, se)],
        "n_risk": nr, "n_event": nk,
        "naive_one_minus_km": naive,
        "naive_overstates_by": (naive[-1] - F[-1]) if naive and F else 0.0,
        "overall_survival_at_end": surv,
        "cause": k, "n": len(t),
        "method": "Aalen-Johansen cumulative incidence; the naive curve "
                  "censors the competing events and overstates"})


def finegray(time, cause, X, code=1, max_iter=50, tol=1e-9):
    """Fine and Gray subdistribution hazard model.

    Fine and Gray (1999, JASA 94:496-509).  A Cox model is fitted on the
    SUBDISTRIBUTION risk set: a subject who fails of a competing cause
    stays in the risk set past its failure time, carrying the
    inverse-probability-of-censoring weight

        w_i(t) = G(t) / G(T_i),   t > T_i,

    with G the Kaplan-Meier of the CENSORING distribution.

    That is the whole difference from a cause-specific Cox model, and
    the two answer different questions: cause-specific hazards describe
    the rate among those still event-free, the subdistribution hazard
    describes the effect on the cumulative incidence itself.  A
    covariate can raise one and lower the other.

    The weights come from the reverse Kaplan-Meier, which assumes
    censoring is independent of covariates; ``covariate_independent_
    censoring_assumed`` records that this implementation does not fit
    the covariate-dependent version.
    """
    t = _flat(time)
    c = _ints(cause)
    Xm, p = _mat(X)
    if not (len(t) == len(c) == len(Xm)):
        raise ValueError("time, cause and X must agree in length")
    k = int(code)
    n = len(t)
    # reverse KM: censoring as the "event"
    cens = [1 if c[i] == 0 else 0 for i in range(n)]
    gt, gS, _, _, _ = _km_curve(t, cens)

    def G(x):
        best = 1.0
        for u, s in zip(gt, gS):
            if u <= x:
                best = s
            else:
                break
        return best

    ut = sorted(set(t[i] for i in range(n) if c[i] == k))
    if not ut:
        raise ValueError("cause %d does not occur in the data" % k)

    def riskset(u):
        """(index, weight) pairs in the subdistribution risk set at u."""
        out = []
        for i in range(n):
            if t[i] >= u:
                out.append((i, 1.0))
            elif c[i] != 0 and c[i] != k:
                gi = G(t[i])
                out.append((i, (G(u) / gi) if gi > 0 else 0.0))
        return [(i, w) for i, w in out if w > 0]

    beta = [0.0] * p
    for _ in range(int(max_iter)):
        g = [0.0] * p
        Hm = [[0.0] * p for _ in range(p)]
        for u in ut:
            rs = riskset(u)
            ev = [i for i in range(n) if t[i] == u and c[i] == k]
            wt = {i: w * exp(fsum(Xm[i][q] * beta[q] for q in range(p)))
                  for i, w in rs}
            s0 = fsum(wt.values())
            if s0 <= 0:
                continue
            s1 = [fsum(Xm[i][q] * wt[i] for i in wt) for q in range(p)]
            for q in range(p):
                g[q] += fsum(Xm[i][q] for i in ev) - len(ev) * s1[q] / s0
            for a in range(p):
                for bq in range(p):
                    s2 = fsum(Xm[i][a] * Xm[i][bq] * wt[i] for i in wt)
                    Hm[a][bq] += len(ev) * (s2 / s0
                                            - s1[a] * s1[bq] / (s0 * s0))
        step = _solve(Hm, g)
        beta = [beta[q] + step[q] for q in range(p)]
        if max(abs(v) for v in step) < tol:
            break
    V = _inv(Hm)
    se = [sqrt(V[q][q]) if V[q][q] > 0 else float("nan") for q in range(p)]
    z = [beta[q] / se[q] if se[q] > 0 else 0.0 for q in range(p)]
    return RichResult(payload={
        "coef": beta, "se": se, "z": z,
        "p_value": [2.0 * _norm_sf(abs(v)) for v in z],
        "subdistribution_hazard_ratio": [exp(v) for v in beta],
        "vcov": V, "n": n, "n_events": sum(1 for v in c if v == k),
        "n_competing": sum(1 for v in c if v != 0 and v != k),
        "cause": k,
        "covariate_independent_censoring_assumed": True,
        "differs_from_cause_specific": True,
        "method": "Fine and Gray (1999) subdistribution hazard, IPCW "
                  "risk set"})


def _solve(A, b):
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("the information matrix is singular")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for j in range(c, n + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][n] / M[i][i] for i in range(n)]


def _inv(A):
    n = len(A)
    cols = [_solve(A, [1.0 if i == j else 0.0 for i in range(n)])
            for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


# ------------------------------------------- truncation, landmark, interval

def ltkm(entry, time, event, alpha=0.05):
    """Kaplan-Meier with left truncation (delayed entry).

    The only change from the ordinary estimator is the risk set:

        n_i = # { j : entry_j < t_i <= time_j },

    so a subject contributes only over the window it was actually under
    observation.  Ignoring the truncation and using entry = 0 counts
    subjects as at risk before they were enrolled, which biases the
    early survival upward -- exactly the immortal-time bias that makes
    prevalent-cohort studies look protective.

    Both curves are returned so the size of that bias is visible.  A
    risk set that empties at some event time makes S undefined beyond
    it; that is reported rather than silently truncating the curve.
    """
    en = _flat(entry)
    t, e = _check(time, event)
    if len(en) != len(t):
        raise ValueError("entry and time must have the same length")
    if any(en[i] >= t[i] for i in range(len(t))):
        raise ValueError("every entry time must be strictly before its "
                         "follow-up time")
    ut = sorted(set(t[i] for i in range(len(t)) if e[i] == 1))
    S, se, nr, ne = [], [], [], []
    surv, v, empty = 1.0, 0.0, []
    for u in ut:
        n_i = sum(1 for j in range(len(t)) if en[j] < u <= t[j])
        d_i = sum(1 for j in range(len(t)) if t[j] == u and e[j] == 1)
        if n_i == 0:
            empty.append(u)
            S.append(surv)
            se.append(sqrt(v) * surv)
            nr.append(0)
            ne.append(d_i)
            continue
        surv *= (1.0 - d_i / n_i)
        if n_i > d_i:
            v += d_i / (n_i * (n_i - d_i))
        S.append(surv)
        se.append(surv * sqrt(v))
        nr.append(n_i)
        ne.append(d_i)
    naive = _km_curve(t, e)[1]
    z = _norm_q(1.0 - alpha / 2.0)
    return RichResult(payload={
        "time": ut, "surv": S, "se": se, "n_risk": nr, "n_event": ne,
        "lower": [max(0.0, a - z * b) for a, b in zip(S, se)],
        "upper": [min(1.0, a + z * b) for a, b in zip(S, se)],
        "ignoring_truncation": naive,
        "max_difference": max((abs(a - b) for a, b in zip(S, naive)),
                              default=0.0),
        "empty_risk_sets": empty, "n": len(t), "n_events": sum(e),
        "method": "Kaplan-Meier with the risk set restricted to "
                  "entry < t <= time"})


def landmark(time, event, landmark_time, X=None, group=None, alpha=0.05):
    """Landmark analysis: condition on surviving to a fixed time.

    Subjects who fail or are censored before the landmark are DROPPED,
    the clock is reset to the landmark, and the analysis proceeds on the
    survivors.

    This is the standard remedy for immortal-time bias when the exposure
    is only known after baseline -- classifying subjects by something
    that could not have happened unless they survived long enough makes
    the exposed group look protected for no reason other than that they
    lived. The landmark removes the guarantee period from both arms.

    The cost is stated rather than hidden: ``n_dropped`` is how much
    data the landmark discards, and the estimate is conditional on
    surviving to it, so it does not describe the whole cohort.
    """
    t, e = _check(time, event)
    lm = float(landmark_time)
    if lm <= 0:
        raise ValueError("the landmark must be positive")
    keep = [i for i in range(len(t)) if t[i] > lm]
    if len(keep) < 2:
        raise ValueError("the landmark leaves %d subjects; it is past the "
                         "bulk of the follow-up" % len(keep))
    tt = [t[i] - lm for i in keep]
    ee = [e[i] for i in keep]
    out = {"landmark": lm, "n_original": len(t), "n_retained": len(keep),
           "n_dropped": len(t) - len(keep), "kept_index": keep,
           "time": tt, "event": ee,
           "conditional_on_surviving_to_landmark": True,
           "method": "landmark analysis; the clock is reset at the "
                     "landmark and earlier subjects are dropped"}
    km = _km_curve(tt, ee)
    out["km_time"] = km[0]
    out["km_surv"] = km[1]
    if group is not None:
        g = list(group)
        if len(g) != len(t):
            raise ValueError("group must have one entry per observation")
        out["group"] = [g[i] for i in keep]
        levels = sorted(set(out["group"]), key=str)
        out["levels"] = levels
        curves = {}
        for lv in levels:
            idx = [j for j, v in enumerate(out["group"]) if v == lv]
            if len(idx) >= 2:
                c = _km_curve([tt[j] for j in idx], [ee[j] for j in idx])
                curves[str(lv)] = {"time": c[0], "surv": c[1]}
        out["by_group"] = curves
    if X is not None:
        Xm, p = _mat(X)
        if len(Xm) != len(t):
            raise ValueError("X must have one row per observation")
        out["X"] = [Xm[i] for i in keep]
        out["p"] = p
    return RichResult(payload=out)


def turnbull(left, right, max_iter=1000, tol=1e-10):
    """Turnbull's NPMLE for interval-censored data.

    Turnbull (1976, JRSS-B 38:290-295).  Each observation is an interval
    (L_i, R_i] known to contain the event; R = inf marks right
    censoring.  The estimator puts mass only on the "Turnbull
    intervals" -- the maximal intervals [q_j, p_j] whose left endpoint
    is some L and right endpoint the next R, with no other endpoint
    between -- and finds the masses by self-consistency:

        p_j^(new) = (1/n) sum_i  p_j 1{I_j subset (L_i, R_i]}
                                 / sum_l p_l 1{I_l subset (L_i, R_i]}.

    That is an EM algorithm and converges monotonically in likelihood,
    but only to a LOCAL maximum in general, and the NPMLE is not unique
    inside a Turnbull interval -- the estimator says the mass is in
    there but not where.  The survival curve is therefore undefined
    (an interval, not a value) across each such gap, which is returned
    as ``ambiguous_intervals`` rather than being interpolated away.

    Exact observations are entered as L = R - epsilon, or as a
    degenerate interval; a zero-width interval is accepted.
    """
    L = _flat(left)
    R = [float(v) if v is not None else inf for v in right]
    if len(L) != len(R):
        raise ValueError("left and right must have the same length")
    n = len(L)
    if not n:
        raise ValueError("need at least one interval")
    if any(L[i] > R[i] for i in range(n)):
        raise ValueError("every left endpoint must not exceed its right")
    lefts = sorted(set(L))
    rights = sorted(set(r for r in R if isfinite(r)))
    inner = []
    for q in lefts:
        cand = [r for r in rights if r >= q]
        if not cand:
            continue
        p_end = min(cand)
        # maximal: no other endpoint strictly inside (q, p_end)
        if any(q < x < p_end for x in lefts) or \
           any(q < x < p_end for x in rights):
            continue
        inner.append((q, p_end))
    inner = sorted(set(inner))
    if not inner:
        raise ValueError("no Turnbull interval could be formed; every "
                         "observation may be right-censored")
    m = len(inner)
    alpha = [[1.0 if (L[i] <= q and p <= R[i]) else 0.0
              for (q, p) in inner] for i in range(n)]
    if any(not any(row) for row in alpha):
        raise ValueError("an observation is compatible with no Turnbull "
                         "interval; check the endpoints")
    p_mass = [1.0 / m] * m
    it, change = 0, inf
    for it in range(1, int(max_iter) + 1):
        new = [0.0] * m
        for i in range(n):
            denom = fsum(alpha[i][j] * p_mass[j] for j in range(m))
            if denom <= 0:
                continue
            for j in range(m):
                if alpha[i][j]:
                    new[j] += alpha[i][j] * p_mass[j] / denom
        new = [v / n for v in new]
        change = max(abs(a - b) for a, b in zip(new, p_mass))
        p_mass = new
        if change < tol:
            break
    surv, s = [], 1.0
    for j in range(m):
        s -= p_mass[j]
        surv.append(max(0.0, s))
    ll = fsum(log(fsum(alpha[i][j] * p_mass[j] for j in range(m)))
              for i in range(n)
              if fsum(alpha[i][j] * p_mass[j] for j in range(m)) > 0)
    return RichResult(payload={
        "intervals": inner, "mass": p_mass, "surv": surv,
        "loglik": ll, "iterations": it, "change": change,
        "converged": change < tol,
        "ambiguous_intervals": [iv for iv, pm in zip(inner, p_mass)
                                if pm > 1e-8],
        "n": n, "n_intervals": m,
        "npmle_not_unique_within_intervals": True,
        "method": "Turnbull (1976) self-consistency / EM"})


# --------------------------------------------------------- parametric fits

_DISTS = ("exponential", "weibull", "lognormal", "loglogistic")


def _logsf_logpdf(dist, y, mu, logsig):
    """(log S, log f) for a log-location-scale family at log-time y."""
    sig = exp(logsig)
    z = (y - mu) / sig
    if dist == "weibull":
        # extreme-value: S = exp(-e^z), f = e^z exp(-e^z) / (sig T)
        ez = exp(z) if z < 700 else float("inf")
        return -ez, (z - ez - logsig)
    if dist == "exponential":
        ez = exp(z) if z < 700 else float("inf")
        return -ez, (z - ez - logsig)
    if dist == "lognormal":
        from ._sci_core import erfc

        S = 0.5 * erfc(z / sqrt(2.0))
        S = max(S, 1e-300)
        lf = -0.5 * z * z - 0.5 * log(2.0 * 3.141592653589793) - logsig
        return log(S), lf
    if dist == "loglogistic":
        # logistic in z: S = 1/(1+e^z)
        if z > 700:
            return -z, (-z - logsig)
        ez = exp(z)
        S = 1.0 / (1.0 + ez)
        return log(S), (z - 2.0 * log(1.0 + ez) - logsig)
    raise ValueError("unknown distribution %r; known: %s"
                     % (dist, ", ".join(_DISTS)))


def _fit_lls(dist, time, event, X=None):
    """ML fit of a log-location-scale model with right censoring."""
    t, e = _check(time, event)
    if any(v <= 0 for v in t):
        raise ValueError("a log-location-scale model needs positive times")
    y = [log(v) for v in t]
    n = len(y)
    if X is None:
        Xm, p = [[1.0] for _ in range(n)], 1
    else:
        Xm, p = _mat(X)
        if len(Xm) != n:
            raise ValueError("X must have one row per observation")
        Xm = [[1.0] + row for row in Xm]
        p += 1
    fixed_scale = (dist == "exponential")

    def nll(theta):
        beta = theta[:p]
        ls = 0.0 if fixed_scale else theta[p]
        tot = 0.0
        for i in range(n):
            mu = fsum(Xm[i][k] * beta[k] for k in range(p))
            lS, lf = _logsf_logpdf(dist, y[i], mu, ls)
            v = lf if e[i] else lS
            if not isfinite(v):
                return 1e300
            tot -= v
        return tot

    mu0 = fsum(y) / n
    var0 = fsum((v - mu0) ** 2 for v in y) / max(n - 1, 1)
    x0 = [mu0] + [0.0] * (p - 1)
    if not fixed_scale:
        x0 = x0 + [0.5 * log(max(var0, 1e-6))]
    res = _sc.minimize(nll, x0, method="nelder-mead")
    theta = list(res["x"]) if isinstance(res, dict) else list(res.x)
    ll = -nll(theta)
    k = len(theta)
    return {"dist": dist, "coef": theta[:p],
            "log_scale": (0.0 if fixed_scale else theta[p]),
            "scale": (1.0 if fixed_scale else exp(theta[p])),
            "loglik": ll, "n_par": k, "n": n, "n_events": sum(e),
            "aic": 2.0 * k - 2.0 * ll,
            "bic": k * log(n) - 2.0 * ll,
            "fixed_scale": fixed_scale}


def parasurv(time, event, dist="weibull"):
    """Parametric survival fit, no covariates.

    Maximum likelihood for one of the log-location-scale families --
    exponential, Weibull, log-normal, log-logistic -- with right
    censoring, the likelihood being

        prod_i f(t_i)^delta_i S(t_i)^(1 - delta_i).

    Censored observations enter through S, not f, and that is the whole
    of the censoring handling: dropping them or treating them as events
    biases the fit in opposite directions.

    The exponential is the Weibull with the scale fixed at 1, so its
    log-likelihood is never higher; comparing the two by a likelihood
    ratio on 1 degree of freedom is the standard test of whether the
    hazard is really constant, and it is returned.
    """
    if dist not in _DISTS:
        raise ValueError("unknown distribution %r; known: %s"
                         % (dist, ", ".join(_DISTS)))
    fit = _fit_lls(dist, time, event)
    out = dict(fit)
    out["intercept"] = fit["coef"][0]
    if dist in ("weibull", "exponential"):
        out["weibull_shape"] = 1.0 / fit["scale"]
        out["weibull_scale"] = exp(fit["coef"][0])
    if dist == "weibull":
        ex = _fit_lls("exponential", time, event)
        lr = 2.0 * (fit["loglik"] - ex["loglik"])
        out["lr_vs_exponential"] = lr
        out["lr_p_value"] = _chi2_sf(max(lr, 0.0), 1)
        out["constant_hazard_rejected"] = out["lr_p_value"] < 0.05
    out["method"] = ("maximum likelihood for a log-location-scale family "
                     "with right censoring")
    return RichResult(payload=out)


def aftfit(time, event, X, dist="weibull", alpha=0.05):
    """Accelerated failure time model.

        log T = x' beta + sigma W,

    with W from the family's error distribution.  The coefficients act
    MULTIPLICATIVELY ON TIME: exp(beta_j) is the factor by which a unit
    of x_j stretches survival, so a positive coefficient means longer
    life.

    That is the opposite sign convention to a Cox model, where a
    positive coefficient means higher hazard and SHORTER life.  Reading
    an AFT coefficient as a log hazard ratio flips the direction of
    every effect; ``time_ratio`` is named to make the scale explicit.

    For the Weibull family alone the AFT and proportional-hazards
    parameterizations describe the same model, related by
    beta_PH = -beta_AFT / sigma; that conversion is returned for
    Weibull and exponential and omitted otherwise, since it does not
    hold for log-normal or log-logistic.
    """
    if dist not in _DISTS:
        raise ValueError("unknown distribution %r; known: %s"
                         % (dist, ", ".join(_DISTS)))
    fit = _fit_lls(dist, time, event, X)
    beta = fit["coef"]
    out = dict(fit)
    out["intercept"] = beta[0]
    out["beta"] = beta[1:]
    out["time_ratio"] = [exp(v) for v in beta[1:]]
    out["positive_coef_means_longer_survival"] = True
    if dist in ("weibull", "exponential"):
        sig = fit["scale"]
        out["ph_coef"] = [-v / sig for v in beta[1:]]
        out["hazard_ratio"] = [exp(-v / sig) for v in beta[1:]]
        out["ph_equivalent"] = True
    else:
        out["ph_equivalent"] = False
    out["method"] = ("accelerated failure time, log T = x'beta + sigma W")
    return RichResult(payload=out)


def paracompare(time, event, X=None, dists=None):
    """Compare parametric survival families on the same data.

    Each family is fitted by maximum likelihood and ranked by AIC, with
    BIC alongside.  Because the four families are NOT nested (except
    exponential inside Weibull), a likelihood ratio test does not apply
    between most pairs -- AIC is the comparison that does, and the
    exponential-in-Weibull LR test is reported separately because it is
    the one case where a formal test exists.

    A family that fails to converge is reported with its error rather
    than dropped, so the ranking cannot silently be over a subset.

    The best AIC is not evidence the winner FITS; check the Cox-Snell
    residuals of the chosen model with :func:`coxsnell` before relying
    on it.
    """
    names = list(dists) if dists is not None else list(_DISTS)
    fits, errs = {}, {}
    for d in names:
        try:
            fits[d] = (aftfit(time, event, X, dist=d) if X is not None
                       else parasurv(time, event, dist=d))
        except Exception as exc:                       # noqa: BLE001
            errs[d] = "%s: %s" % (type(exc).__name__, exc)
    if not fits:
        raise ValueError("no family could be fitted: %r" % errs)
    rows = [{"dist": d, "loglik": f["loglik"], "aic": f["aic"],
             "bic": f["bic"], "n_par": f["n_par"]}
            for d, f in fits.items()]
    rows.sort(key=lambda r: r["aic"])
    out = {"table": rows, "best_aic": rows[0]["dist"],
           "best_bic": min(rows, key=lambda r: r["bic"])["dist"],
           "fits": fits, "failed": errs,
           "families_not_nested": True,
           "aic_is_not_goodness_of_fit": True,
           "method": "AIC / BIC comparison of parametric survival families"}
    if "weibull" in fits and "exponential" in fits:
        lr = 2.0 * (fits["weibull"]["loglik"] - fits["exponential"]["loglik"])
        out["lr_weibull_vs_exponential"] = lr
        out["lr_p_value"] = _chi2_sf(max(lr, 0.0), 1)
    return RichResult(payload=out)
