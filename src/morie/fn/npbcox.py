# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric Bayes Cox model with a gamma-process baseline."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["np_bayes_cox"]


def np_bayes_cox(time, event, X, c=1.0, lam0=None, n_iter=50, tol=1e-12):
    """Cox regression with a gamma-process prior on the cumulative hazard.

    Kalbfleisch put a gamma process ``H ~ GP(c, H_0)`` on the baseline
    cumulative hazard and left the regression coefficient
    unconstrained.  The posterior-mean increment at an observed event
    time is the prior increment and the observed count pooled by their
    weights,

        dH(t_k) = (c dH_0(t_k) + dN_k) / (c + sum_{j in R_k} exp(x_j'b)),

    with ``R_k`` the risk set at ``t_k``.  As ``c -> 0`` the prior
    washes out and this is exactly the Breslow estimator; with
    ``b = 0`` as well it is the Nelson-Aalen estimator.  Both limits
    are asserted as anchors.

    ``b`` is obtained by Newton-Raphson on the Breslow partial
    log-likelihood,

        l(b) = sum_k [ sum_{i in D_k} x_i'b - d_k log sum_{j in R_k} e^{x_j'b} ],

    which is concave, so the Newton step is taken against the NEGATIVE
    Hessian -- the observed information -- and no line search is needed.

    Determinism: fixed iteration cap plus a gradient-norm rule; no
    sampling.

    Parameters
    ----------
    time : array-like, shape (n,)
        Observation times.
    event : array-like, shape (n,)
        1 = event, 0 = right-censored.
    X : array-like, shape (n, p)
        Covariates, no intercept (a Cox model has none).
    c : float, default 1.0
        Gamma-process prior weight; ``c = 0`` gives Breslow.
    lam0 : float or None
        Constant base hazard rate defining ``dH_0``; ``1 / mean(time)``
        if ``None``.
    n_iter : int, default 50
        Newton iteration cap.
    tol : float, default 1e-12
        Gradient-norm stopping rule.

    Returns
    -------
    RichResult
        ``beta``, ``se``, ``estimate`` (first coefficient),
        ``loglik``, ``times`` (distinct event times), ``dH``, ``H``
        (posterior-mean cumulative hazard), ``S`` (survival),
        ``iterations``, ``converged``, ``n``, ``n_events``.

    References
    ----------
    Kalbfleisch, J. D. (1978).  Non-parametric Bayesian analysis of
    survival time data.  Journal of the Royal Statistical Society
    Series B, 40(2), 214--221.  Cox, D. R. (1972).  Regression models
    and life-tables.  JRSS B, 34(2), 187--220.
    """
    t = C.vec(time)
    n = len(t)
    if n == 0:
        raise ValueError("np_bayes_cox: time is empty")
    d = [1.0 if float(v) != 0.0 else 0.0 for v in event]
    if len(d) != n:
        raise ValueError("np_bayes_cox: time and event have different lengths")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("np_bayes_cox: X and time have different lengths")
    p = len(Xm[0])
    cc = float(c)
    if cc < 0.0:
        raise ValueError("np_bayes_cox: c must be non-negative")
    l0 = 1.0 / C.mean(t) if lam0 is None else float(lam0)

    ev = sorted({t[i] for i in range(n) if d[i] == 1.0})
    if not ev:
        raise ValueError("np_bayes_cox: no events")

    def parts(b):
        eta = [sum(Xm[i][k] * b[k] for k in range(p)) for i in range(n)]
        w = [math.exp(e) for e in eta]
        ll = 0.0
        g = [0.0] * p
        H = [[0.0] * p for _ in range(p)]
        for tk in ev:
            R = [i for i in range(n) if t[i] >= tk]
            D = [i for i in range(n) if t[i] == tk and d[i] == 1.0]
            s0 = sum(w[i] for i in R)
            s1 = [sum(w[i] * Xm[i][k] for i in R) for k in range(p)]
            dk = float(len(D))
            for i in D:
                ll += eta[i]
            ll -= dk * math.log(s0)
            for k in range(p):
                g[k] += sum(Xm[i][k] for i in D) - dk * s1[k] / s0
                for m in range(p):
                    s2 = sum(w[i] * Xm[i][k] * Xm[i][m] for i in R)
                    H[k][m] += dk * (s2 / s0 - s1[k] * s1[m] / (s0 * s0))
        return ll, g, H, w

    beta = [0.0] * p
    it = 0
    ll, g, Hm, w = parts(beta)
    gn = math.sqrt(sum(v * v for v in g))
    while it < int(n_iter) and gn > float(tol):
        step = C.solvev(Hm, g)
        beta = [beta[k] + step[k] for k in range(p)]
        it += 1
        ll, g, Hm, w = parts(beta)
        gn = math.sqrt(sum(v * v for v in g))
    # A covariate collinear with the baseline (a constant, say) leaves the
    # information matrix singular. That is a real answer -- the coefficient
    # is not identified -- not a reason to abort, so the standard errors
    # come back NaN and the estimates stand.
    try:
        cov = C.inv(Hm)
        se = [math.sqrt(cov[k][k]) if cov[k][k] > 0.0 else float("nan")
              for k in range(p)]
    except ValueError:
        se = [float("nan")] * p

    dH = []
    Hcum = []
    S = []
    acc = 0.0
    surv = 1.0
    prev = 0.0
    for tk in ev:
        risk = sum(w[i] for i in range(n) if t[i] >= tk)
        dk = float(sum(1 for i in range(n) if t[i] == tk and d[i] == 1.0))
        dh0 = l0 * (tk - prev)
        prev = tk
        inc = (cc * dh0 + dk) / (cc + risk)
        dH.append(inc)
        acc += inc
        Hcum.append(acc)
        f = 1.0 - inc
        if f < 1e-15:
            f = 1e-15
        surv *= f
        S.append(surv)
    return RichResult(payload={
        "beta": beta, "se": se, "estimate": beta[0], "loglik": ll,
        "times": ev, "dH": dH, "H": Hcum, "S": S,
        "grad_norm": gn, "iterations": it,
        "converged": 1.0 if gn <= float(tol) else 0.0,
        "n": n, "n_events": int(sum(d)), "c": cc, "lam0": l0,
        "method": "Cox model with a gamma-process baseline (Kalbfleisch 1978)"})


def cheatsheet():
    return "npbcox: Cox regression with a gamma-process baseline hazard"


npbayescox = np_bayes_cox
