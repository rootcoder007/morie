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


def _transform(times, e_times, how):
    if how == "identity":
        return list(times)
    if how == "log":
        return [log(v) for v in times]
    if how == "rank":
        s = sorted(range(len(times)), key=lambda i: times[i])
        r = [0.0] * len(times)
        for rank, i in enumerate(s):
            r[i] = float(rank + 1)
        return r
    # "km": 1 - KM estimate over the event times, the default in most software
    n = len(e_times)
    surv, out = 1.0, []
    at_risk = n
    for tt in times:
        d = sum(1 for v in e_times if v == tt)
        nr = sum(1 for v in e_times if v >= tt)
        out.append(1.0 - surv)
        if nr > 0:
            surv *= 1.0 - d / nr
    return out


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
              \operatorname{Var}(U)=\sum_j (g_j-\bar g)^2 V_j ,

    giving :math:`U^{\top}\operatorname{Var}(U)^{-1}U\sim\chi^2_p`.
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
    g = _transform(times, e_times, transform)
    gbar = sum(g) / d
    gc = [v - gbar for v in g]

    scaled = [[beta[k] + d * sum(Vbl[k][m] * res[j][m] for m in range(p)) for k in range(p)]
              for j in range(d)]
    U = [sum(gc[j] * res[j][k] for j in range(d)) for k in range(p)]
    VU = [[sum(gc[j] * gc[j] * var[j][a][b] for j in range(d)) for b in range(p)]
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
