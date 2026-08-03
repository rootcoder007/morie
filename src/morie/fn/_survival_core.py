# morie.fn -- shared core (rootcoder007/morie)
"""Survival analysis: Kaplan-Meier, Nelson-Aalen, log-rank and Cox.

Verified numerically against R's ``survival`` package, which is the
reference implementation for this material:

    kaplan_meier        survival::survfit
    nelson_aalen        survival::survfit(type = "fh")
    logrank_test        survival::survdiff
    cox_ph              survival::coxph (Efron ties, the default)
    concordance_index   survival::concordance

Censoring is the whole difficulty here: a subject still alive at the
end of follow-up contributes the information that it survived that
long, and discarding it -- or treating it as an event -- biases every
estimate.  Each routine below handles it explicitly.

No external numeric dependency.
"""

import math

__all__ = [
    "kaplan_meier", "nelson_aalen", "logrank_test", "cox_ph",
    "cox_partial_loglik", "concordance_index",
]


def _flat(v):
    return [float(t) for t in v]


def _risk_table(time, event):
    """Distinct event times with the number at risk and failing."""
    t = _flat(time)
    d = [int(v) for v in event]
    n = len(t)
    times = sorted(set(t[i] for i in range(n) if d[i] == 1))
    out = []
    for u in times:
        at_risk = sum(1 for i in range(n) if t[i] >= u)
        events = sum(1 for i in range(n) if t[i] == u and d[i] == 1)
        out.append((u, at_risk, events))
    return out


def kaplan_meier(time, event, alpha=0.05):
    """Kaplan-Meier product-limit estimator of the survival function.

        S(t) = prod_{t_i <= t} (1 - d_i / n_i)

    with Greenwood's variance

        Var(S) = S^2 sum d_i / (n_i (n_i - d_i))

    and a log-log confidence interval, which is ``survival``'s
    default
    because it cannot stray outside [0, 1] the way a plain symmetric
    interval can.  Censored observations leave the risk set without
    contributing an event, which is exactly how they carry information.
    """
    t = _flat(time)
    d = [int(v) for v in event]
    if len(t) != len(d):
        raise ValueError("time and event must have the same length")
    if any(v not in (0, 1) for v in d):
        raise ValueError("event must be 0 (censored) or 1 (event)")
    tab = _risk_table(t, d)
    surv, var_sum = 1.0, 0.0
    times, S, se, lo, hi, nrisk, nevent = [], [], [], [], [], [], []
    se_ch = []
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else \
        _norm_q(1 - alpha / 2)
    for (u, n_i, d_i) in tab:
        surv *= (1.0 - d_i / n_i)
        if n_i > d_i:
            var_sum += d_i / (n_i * (n_i - d_i))
        # two standard errors are in circulation and they differ by a
        # factor of S: survival::survfit's `std.err` is the error of the
        # CUMULATIVE HAZARD, sqrt(Greenwood sum), while the error of
        # S(t) itself carries the extra S.  Both are returned, named.
        times.append(u)
        S.append(surv)
        se.append(surv * math.sqrt(var_sum))
        se_ch.append(math.sqrt(var_sum))
        nrisk.append(n_i)
        nevent.append(d_i)
        if 0.0 < surv < 1.0:
            # log-log interval, as survival::survfit conf.type="log-log"
            ll = math.log(-math.log(surv))
            sd = math.sqrt(var_sum) / abs(math.log(surv))
            lo.append(math.exp(-math.exp(ll + z * sd)))
            hi.append(math.exp(-math.exp(ll - z * sd)))
        else:
            lo.append(surv)
            hi.append(surv)
    return {"time": times, "surv": S, "se": se,
            "se_cumhaz": se_ch, "lower": lo,
            "upper": hi, "n_risk": nrisk, "n_event": nevent,
            "n": len(t), "n_events": sum(d),
            "method": "Kaplan-Meier product-limit estimator"}


def _norm_q(p):
    """Standard normal quantile (Acklam, refined by one Halley step)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01,
          2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    elif p <= 1 - pl:
        q = p - 0.5
        r = q * q
        z = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        z = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    e = 0.5 * math.erfc(-z / math.sqrt(2)) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(z * z / 2)
    return z - u / (1 + z * u / 2)


def _chi2_sf(x, df):
    if x <= 0:
        return 1.0
    a, xx = df / 2.0, x / 2.0
    if xx < a + 1.0:
        term = 1.0 / a
        s, nn = term, a
        for _ in range(1000):
            nn += 1.0
            term *= xx / nn
            s += term
            if abs(term) < abs(s) * 1e-16:
                break
        return 1.0 - s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
    b = xx + 1.0 - a
    c, d = 1e300, 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h * math.exp(-xx + a * math.log(xx) - math.lgamma(a))


def nelson_aalen(time, event):
    """Nelson-Aalen estimator of the cumulative hazard.

        H(t) = sum_{t_i <= t} d_i / n_i,   Var = sum d_i / n_i^2

    Better behaved than -log(KM) in small samples, and the natural
    scale on which to judge proportional hazards: parallel cumulative
    hazards on the log scale are what the Cox model assumes.
    """
    tab = _risk_table(time, event)
    H, V = 0.0, 0.0
    times, ch, se, surv = [], [], [], []
    for (u, n_i, d_i) in tab:
        H += d_i / n_i
        V += d_i / (n_i * n_i)
        times.append(u)
        ch.append(H)
        se.append(math.sqrt(V))
        surv.append(math.exp(-H))
    return {"time": times, "cumhaz": ch, "se": se, "surv": surv,
            "method": "Nelson-Aalen cumulative hazard"}


def logrank_test(time, event, group):
    """Log-rank test comparing survival between groups.

    At each event time the observed failures in a group are compared
    with the number expected if the groups shared a hazard; the
    statistic is the sum of those differences, standardised by the
    hypergeometric variance.  Equivalent to ``survival::survdiff``
    with the default ``rho = 0``.
    """
    t = _flat(time)
    d = [int(v) for v in event]
    g = list(group)
    if not (len(t) == len(d) == len(g)):
        raise ValueError("time, event and group must have equal length")
    levels = sorted(set(g))
    k = len(levels)
    if k < 2:
        raise ValueError("need at least 2 groups")
    obs = [0.0] * k
    exp = [0.0] * k
    V = [[0.0] * k for _ in range(k)]
    for u in sorted(set(t[i] for i in range(len(t)) if d[i] == 1)):
        n_i = sum(1 for i in range(len(t)) if t[i] >= u)
        d_i = sum(1 for i in range(len(t)) if t[i] == u and d[i] == 1)
        nj = [sum(1 for i in range(len(t))
                  if t[i] >= u and g[i] == levels[j]) for j in range(k)]
        dj = [sum(1 for i in range(len(t)) if t[i] == u and d[i] == 1
                  and g[i] == levels[j]) for j in range(k)]
        for j in range(k):
            obs[j] += dj[j]
            exp[j] += d_i * nj[j] / n_i
        if n_i > 1:
            f = d_i * (n_i - d_i) / (n_i - 1.0)
            for a in range(k):
                for b in range(k):
                    ind = 1.0 if a == b else 0.0
                    V[a][b] += f * (ind * nj[a] / n_i
                                    - nj[a] * nj[b] / (n_i * n_i))
    # drop one group for identifiability, then quadratic form
    m = k - 1
    diff = [obs[j] - exp[j] for j in range(m)]
    A = [[V[a][b] for b in range(m)] for a in range(m)]
    sol = _solve_lin(A, diff)
    stat = sum(diff[j] * sol[j] for j in range(m))
    return {"statistic": stat, "df": m, "p_value": _chi2_sf(stat, m),
            "observed": obs, "expected": exp, "groups": levels,
            "method": "log-rank test"}


def _solve_lin(A, b):
    n = len(A)
    M = [[float(A[i][j]) for j in range(n)] + [float(b[i])]
         for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-300:
            raise ValueError("singular system")
        M[c], M[piv] = M[piv], M[c]
        dv = M[c][c]
        M[c] = [v / dv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [M[r][q] - f * M[c][q] for q in range(n + 1)]
    return [M[i][n] for i in range(n)]


def _mat(X):
    rows = list(X)
    if rows and not isinstance(rows[0], (list, tuple)):
        return [[float(v)] for v in rows]
    return [[float(v) for v in r] for r in rows]


def cox_partial_loglik(time, event, X, beta, ties="efron"):
    """Log partial likelihood of the Cox model.

    Efron's approximation for tied event times is used by default,
    matching ``survival::coxph``; it is markedly more accurate than
    Breslow's when ties are common, which in practice they are because
    follow-up is recorded in whole days or months.
    """
    t = _flat(time)
    d = [int(v) for v in event]
    Xm = _mat(X)
    b = _flat(beta)
    n = len(t)
    eta = [sum(Xm[i][j] * b[j] for j in range(len(b))) for i in range(n)]
    ll = 0.0
    for u in sorted(set(t[i] for i in range(n) if d[i] == 1)):
        risk = [i for i in range(n) if t[i] >= u]
        died = [i for i in range(n) if t[i] == u and d[i] == 1]
        s_risk = sum(math.exp(eta[i]) for i in risk)
        s_died = sum(math.exp(eta[i]) for i in died)
        m = len(died)
        for i in died:
            ll += eta[i]
        if ties == "breslow":
            ll -= m * math.log(s_risk)
        else:
            for r in range(m):
                ll -= math.log(s_risk - r * s_died / m)
    return ll


def cox_ph(time, event, X, ties="efron", max_iter=50, tol=1e-9):
    """Cox proportional-hazards regression by Newton-Raphson.

        lambda(t | x) = lambda_0(t) exp(x' beta)

    The baseline hazard is left completely unspecified -- that is the
    point of the model -- so beta is estimated from the partial
    likelihood alone.  exp(beta) is a hazard ratio: the multiplicative
    effect on the instantaneous risk of failure, constant over time,
    which is the assumption worth checking before believing the answer.

    Efron's tie handling by default, as ``survival::coxph``.
    """
    t = _flat(time)
    d = [int(v) for v in event]
    Xm = _mat(X)
    n = len(t)
    if not (len(d) == len(Xm) == n):
        raise ValueError("time, event and X must have the same length")
    if sum(d) == 0:
        raise ValueError("no events: the partial likelihood is empty")
    p = len(Xm[0])
    beta = [0.0] * p
    ev_times = sorted(set(t[i] for i in range(n) if d[i] == 1))

    for _ in range(int(max_iter)):
        eta = [sum(Xm[i][j] * beta[j] for j in range(p))
               for i in range(n)]
        w = [math.exp(e) for e in eta]
        grad = [0.0] * p
        H = [[0.0] * p for _ in range(p)]
        for u in ev_times:
            risk = [i for i in range(n) if t[i] >= u]
            died = [i for i in range(n) if t[i] == u and d[i] == 1]
            m = len(died)
            s0r = sum(w[i] for i in risk)
            s1r = [sum(w[i] * Xm[i][a] for i in risk) for a in range(p)]
            s2r = [[sum(w[i] * Xm[i][a] * Xm[i][b] for i in risk)
                    for b in range(p)] for a in range(p)]
            s0d = sum(w[i] for i in died)
            s1d = [sum(w[i] * Xm[i][a] for i in died) for a in range(p)]
            s2d = [[sum(w[i] * Xm[i][a] * Xm[i][b] for i in died)
                    for b in range(p)] for a in range(p)]
            for i in died:
                for a in range(p):
                    grad[a] += Xm[i][a]
            steps = 1 if ties == "breslow" else m
            for r in range(steps):
                frac = 0.0 if ties == "breslow" else r / m
                cnt = m if ties == "breslow" else 1
                s0 = s0r - frac * s0d
                s1 = [s1r[a] - frac * s1d[a] for a in range(p)]
                s2 = [[s2r[a][b] - frac * s2d[a][b] for b in range(p)]
                      for a in range(p)]
                for a in range(p):
                    grad[a] -= cnt * s1[a] / s0
                for a in range(p):
                    for b in range(p):
                        H[a][b] += cnt * (s2[a][b] / s0
                                          - s1[a] * s1[b] / (s0 * s0))
        try:
            step = _solve_lin(H, grad)
        except ValueError:
            break
        beta = [beta[a] + step[a] for a in range(p)]
        if max(abs(s) for s in step) < tol:
            break

    # observed information at the optimum gives the standard errors
    eta = [sum(Xm[i][j] * beta[j] for j in range(p)) for i in range(n)]
    w = [math.exp(e) for e in eta]
    H = [[0.0] * p for _ in range(p)]
    for u in ev_times:
        risk = [i for i in range(n) if t[i] >= u]
        died = [i for i in range(n) if t[i] == u and d[i] == 1]
        m = len(died)
        s0r = sum(w[i] for i in risk)
        s1r = [sum(w[i] * Xm[i][a] for i in risk) for a in range(p)]
        s2r = [[sum(w[i] * Xm[i][a] * Xm[i][b] for i in risk)
                for b in range(p)] for a in range(p)]
        s0d = sum(w[i] for i in died)
        s1d = [sum(w[i] * Xm[i][a] for i in died) for a in range(p)]
        s2d = [[sum(w[i] * Xm[i][a] * Xm[i][b] for i in died)
                for b in range(p)] for a in range(p)]
        steps = 1 if ties == "breslow" else m
        for r in range(steps):
            frac = 0.0 if ties == "breslow" else r / m
            cnt = m if ties == "breslow" else 1
            s0 = s0r - frac * s0d
            s1 = [s1r[a] - frac * s1d[a] for a in range(p)]
            s2 = [[s2r[a][b] - frac * s2d[a][b] for b in range(p)]
                  for a in range(p)]
            for a in range(p):
                for b in range(p):
                    H[a][b] += cnt * (s2[a][b] / s0
                                      - s1[a] * s1[b] / (s0 * s0))
    cols = []
    for j in range(p):
        e = [1.0 if i == j else 0.0 for i in range(p)]
        cols.append(_solve_lin(H, e))
    V = [[cols[b][a] for b in range(p)] for a in range(p)]
    se = [math.sqrt(V[j][j]) for j in range(p)]
    z = [beta[j] / se[j] for j in range(p)]
    pv = [2.0 * (1.0 - 0.5 * math.erfc(-abs(v) / math.sqrt(2)))
          for v in z]
    ll = cox_partial_loglik(t, d, Xm, beta, ties)
    ll0 = cox_partial_loglik(t, d, Xm, [0.0] * p, ties)
    return {"coef": beta, "se": se, "z": z, "p_value": pv,
            "hazard_ratio": [math.exp(v) for v in beta],
            "vcov": V, "loglik": ll, "loglik_null": ll0,
            "lr_statistic": 2.0 * (ll - ll0),
            "lr_p_value": _chi2_sf(2.0 * (ll - ll0), p),
            "n": n, "n_events": sum(d), "ties": ties,
            "method": "Cox proportional-hazards model"}


def concordance_index(time, event, predicted_risk):
    """Harrell's concordance index for censored outcomes.

    Over all comparable pairs -- those whose order is known despite
    censoring -- the proportion in which the subject who failed first
    carried the higher predicted risk, with ties counted as a half.
    0.5 is chance, 1.0 is perfect discrimination.  Matches
    ``survival::concordance``.
    """
    t = _flat(time)
    d = [int(v) for v in event]
    r = _flat(predicted_risk)
    n = len(t)
    if not (len(d) == len(r) == n):
        raise ValueError("all inputs must have the same length")
    conc = disc = tied = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            # a pair is comparable only if the earlier time is an event
            if t[i] < t[j] and d[i] == 1:
                lo, hi = i, j
            elif t[j] < t[i] and d[j] == 1:
                lo, hi = j, i
            elif t[i] == t[j] and d[i] == 1 and d[j] == 1:
                tied += 1.0 if r[i] != r[j] else 0.0
                continue
            else:
                continue
            if r[lo] > r[hi]:
                conc += 1
            elif r[lo] < r[hi]:
                disc += 1
            else:
                tied += 1
    total = conc + disc + tied
    if total == 0:
        raise ValueError("no comparable pairs")
    return {"c_index": (conc + 0.5 * tied) / total,
            "concordant": conc, "discordant": disc, "tied": tied,
            "n_pairs": total,
            "method": "Harrell's concordance index"}
