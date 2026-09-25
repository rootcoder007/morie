"""Scaled Schoenfeld residuals and the Grambsch-Therneau test of proportional hazards."""

from math import exp, log

from . import _array_core as np
from . import _stats_core as stats
from ._richresult import RichResult
from ._survival_core import cox_ph

__all__ = ["scaled_schoenfeld_residual"]

_TRANSFORMS = ("km", "rank", "identity", "log")


def _schoenfeld(t, e, X, beta):
    """Raw Schoenfeld residuals and the risk-set variance at each event time."""
    n, p = len(t), len(X[0])
    order = sorted(range(n), key=lambda i: t[i])
    times, res, var = [], [], []
    for i in order:
        if e[i] != 1:
            continue
        risk = [j for j in range(n) if t[j] >= t[i]]
        w = [exp(sum(X[j][k] * beta[k] for k in range(p))) for j in risk]
        sw = sum(w)
        xbar = [sum(w[m] * X[risk[m]][k] for m in range(len(risk))) / sw for k in range(p)]
        # weighted covariance of the covariates over the risk set: this is
        # the per-event-time contribution to the information matrix, and
        # it is exactly Var(s_j) under the null.
        V = [[sum(w[m] * (X[risk[m]][a] - xbar[a]) * (X[risk[m]][b] - xbar[b])
                  for m in range(len(risk))) / sw for b in range(p)] for a in range(p)]
        times.append(t[i])
        res.append([X[i][k] - xbar[k] for k in range(p)])
        var.append(V)
    return times, res, var


def _transform(times, t_all, e_all, how):
    """g(t) at each event time, as R's survival::cox.zph computes it.

    "km" is 1 - S(t-), the left-continuous Kaplan-Meier estimate over ALL
    subjects, censored ones included, with tied events taken together;
    "rank" ranks all follow-up times with ties averaged. The previous
    version built the Kaplan-Meier risk sets from the event times alone
    and stepped once per tied duplicate.
    """
    if how == "identity":
        return list(times)
    if how == "log":
        return [log(v) for v in times]
    if how == "rank":
        order = sorted(range(len(t_all)), key=lambda i: t_all[i])
        rk = {}
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and t_all[order[j + 1]] == t_all[order[i]]:
                j += 1
            rk[t_all[order[i]]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return [rk[v] for v in times]
    uniq = sorted(set(t_all))
    surv_before = {}
    surv = 1.0
    for u in uniq:
        surv_before[u] = surv
        nr = sum(1 for v in t_all if v >= u)
        d = sum(1 for v, ev in zip(t_all, e_all) if v == u and ev == 1)
        if nr > 0 and d > 0:
            surv *= 1.0 - d / nr
    return [1.0 - surv_before[v] for v in times]


def scaled_schoenfeld_residual(time, event, X, transform="km"):
    r"""Scaled Schoenfeld residuals and the test of proportional hazards.

    The raw Schoenfeld residual at event time :math:`t_j` is the failing
    subject's covariate minus the hazard-weighted risk-set mean,
    :math:`s_j = x_{(j)} - \bar x_j`. Grambsch & Therneau's insight is
    that rescaling by the information turns the residual into a direct
    estimate of the coefficient *at that time*:

    .. math:: s^{*}_j = \hat\beta + d\;\widehat{\operatorname{Var}}(\hat\beta)\,s_j,
              \qquad E[s^{*}_j]\approx\beta(t_j)

    so a plot of :math:`s^*_j` against :math:`g(t_j)` shows the
    coefficient's trajectory, and proportional hazards is the
    hypothesis that the trajectory is flat.

    The accompanying test is the score test of
    :math:`\beta(t)=\beta+\theta\,g(t)` at :math:`\theta=0`. Because
    :math:`\operatorname{Var}(s_j)=V_j`, the risk-set covariance at
    :math:`t_j`, the score and its variance are available in closed
    form:

    .. math:: U=\sum_j (g_j-\bar g)\,s_j, \qquad
              A=\sum_j (g_j-\bar g)^2 V_j, \quad
              C=\sum_j (g_j-\bar g) V_j, \quad
              I=\sum_j V_j ,

    and, because :math:`\hat\beta` is estimated, the variance of the
    score for :math:`\theta` is the efficient one,
    :math:`A - C I^{-1} C^{\top}`, giving
    :math:`U^{\top}(A - C I^{-1} C^{\top})^{-1}U\sim\chi^2_p` and, per
    covariate, :math:`U_k^2 / (A - C I^{-1} C^{\top})_{kk}`. This is the
    test R's ``survival::cox.zph`` (version 3) reports. Using :math:`A`
    alone treats :math:`\hat\beta` as known and understates the
    statistic.
    No numerical optimisation is involved, so the arms agree exactly
    rather than to an optimiser's tolerance.

    A significant result does not mean the covariate is unimportant. It
    means a *single* hazard ratio is the wrong summary; the fix is
    stratification or a time-varying coefficient, not deletion.

    Parameters
    ----------
    time : array-like
        Follow-up times.
    event : array-like of {0, 1}
        1 for an event, 0 for right-censored.
    X : array-like, shape (n, p)
        Covariates.
    transform : {"km", "rank", "identity", "log"}
        The time transform ``g``. ``"km"`` (1 - Kaplan-Meier) is the
        default in most software and is far less leveraged by a long
        tail than ``"identity"``.

    Returns
    -------
    RichResult
        Keys ``scaled`` (n_events x p), ``residuals`` (raw), ``times``,
        ``gtime``, ``beta``, ``vcov``, ``statistic`` (per covariate),
        ``pvalue``, ``global_statistic``, ``global_pvalue``, ``df``.

    Notes
    -----
    ``mean(scaled) == beta`` exactly: the score equations make the raw
    Schoenfeld residuals sum to zero at the MLE, so the rescaling
    leaves the mean at :math:`\hat\beta`. That identity is a cheap
    check that a fit converged.

    References
    ----------
    Schoenfeld, D. (1982). Partial residuals for the proportional
    hazards regression model. *Biometrika*, 69(1), 239-241.
    Grambsch, P. M. & Therneau, T. M. (1994). Proportional hazards
    tests and diagnostics based on weighted residuals. *Biometrika*,
    81(3), 515-526.
    """
    if transform not in _TRANSFORMS:
        raise ValueError("transform must be one of %s" % (_TRANSFORMS,))
    t = [float(v) for v in np.asarray(time, dtype=float).ravel().tolist()]
    e = [float(v) for v in np.asarray(event, dtype=float).ravel().tolist()]
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    Xl = [[float(Xa[i][j]) for j in range(Xa.shape[1])] for i in range(Xa.shape[0])]
    if not (len(t) == len(e) == len(Xl)):
        raise ValueError("time, event and X must agree in length.")
    p = len(Xl[0])
    fit = cox_ph(t, e, Xa)
    beta = [float(v) for v in np.asarray(fit["coef"], dtype=float).ravel().tolist()]
    Vb = np.atleast_2d(np.asarray(fit["vcov"], dtype=float))
    Vbl = [[float(Vb[i][j]) for j in range(p)] for i in range(p)]

    times, res, var = _schoenfeld(t, e, Xl, beta)
    d = len(times)
    if d < 3:
        raise ValueError("need at least 3 events.")
    e_times = [t[i] for i in range(len(t)) if e[i] == 1]
    g = _transform(times, t, e, transform)
    gbar = sum(g) / d
    gc = [v - gbar for v in g]

    scaled = [[beta[k] + d * sum(Vbl[k][m] * res[j][m] for m in range(p)) for k in range(p)]
              for j in range(d)]
    U = [sum(gc[j] * res[j][k] for j in range(d)) for k in range(p)]
    A = [[sum(gc[j] * gc[j] * var[j][a][b] for j in range(d)) for b in range(p)]
         for a in range(p)]
    C = [[sum(gc[j] * var[j][a][b] for j in range(d)) for b in range(p)]
         for a in range(p)]
    Iinf = [[sum(var[j][a][b] for j in range(d)) for b in range(p)] for a in range(p)]
    # efficient variance of the theta-score: beta-hat is estimated too
    IC = [[float(v) for v in col] for col in np.linalg.solve(
        np.asarray(Iinf, dtype=float), np.asarray(C, dtype=float).T).T.tolist()] \
        if p > 1 else [[C[0][0] / Iinf[0][0]]]
    VU = [[A[a][b] - sum(IC[a][m] * C[b][m] for m in range(p)) for b in range(p)]
          for a in range(p)]
    stat, pv = [], []
    for k in range(p):
        vk = VU[k][k]
        s = U[k] * U[k] / vk if vk > 0 else float("nan")
        stat.append(float(s))
        pv.append(float(stats.chi2.sf(s, 1)))
    try:
        x = np.linalg.solve(np.asarray(VU, dtype=float), np.asarray(U, dtype=float))
        gs = float(sum(U[k] * float(x[k]) for k in range(p)))
    except Exception:
        gs = float("nan")
    return RichResult(
        payload={
            "scaled": scaled,
            "residuals": res,
            "times": times,
            "gtime": g,
            "beta": beta,
            "vcov": Vbl,
            "statistic": stat,
            "pvalue": pv,
            "global_statistic": gs,
            "global_pvalue": float(stats.chi2.sf(gs, p)) if gs == gs else float("nan"),
            "df": p,
            "n_events": d,
            "transform": transform,
            "method": "Grambsch-Therneau (1994) scaled Schoenfeld residuals and PH score test",
        }
    )


def cheatsheet():
    return "shscl: scaled Schoenfeld residuals and the Grambsch-Therneau PH test"
