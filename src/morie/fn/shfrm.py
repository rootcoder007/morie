# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared frailty marginal model."""

import math

from . import _array_core as np

from ._richresult import RichResult
from ._recur_core import cox_counting_process

__all__ = ["shfrm", "shared_frailty_marginal"]


def _profile_theta(D, L, theta):
    # gamma-frailty profile log-likelihood terms in theta (Klein 1992)
    a = 1.0 / theta
    tot = 0.0
    for k in range(len(D)):
        tot += (a * math.log(a) - math.lgamma(a)
                + math.lgamma(D[k] + a) - (D[k] + a) * math.log(a + L[k]))
    return tot


def shared_frailty_marginal(time, event, X, cluster, theta=None):
    """
    Shared gamma frailty model with marginal (population-averaged) output.

    Conditional intensity lambda_i(t | w_k) = w_k lambda_0(t)
    exp(beta' x_i) with w_k ~ Gamma(1/theta, 1/theta) shared within
    cluster k (Vaupel, Manton & Stallard 1979). Fitted by EM: the
    E-step frailty posterior mean is (D_k + 1/theta)/(Lambda_k +
    1/theta), the M-step is a Cox fit with offset log w, and theta
    maximises the gamma profile likelihood by golden section.
    Integrating the frailty out gives the MARGINAL survivor
    S_m(t | x) = {1 + theta Lambda_0(t) e^(beta x)}^(-1/theta),
    whose hazard ratio attenuates toward 1 over time -- the marginal
    summary this module reports alongside the conditional beta.

    References: Vaupel, J. W., Manton, K. G. and Stallard, E. (1979),
    Demography 16(3), 439-454; Klein, J. P. (1992), Biometrics 48,
    795-806 (EM algorithm).

    Returns
    -------
    result : RichResult
        Keys: estimate (conditional beta), se, theta, kendall_tau,
        frailty (cluster -> posterior mean), baseline_times,
        baseline_cumhaz, marginal_survivor (callable-free: values of
        S_m at baseline_times for x = 0), loglik, n_outer.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=float)
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape((-1, 1))
    if Xa.shape[0] != t.shape[0]:
        Xa = Xa.T
    n, p = Xa.shape
    cl = list(cluster)
    ks = sorted(set(cl))
    K = len(ks)
    if K < 2:
        raise ValueError("need at least 2 clusters")
    kidx = {k: [i for i in range(n) if cl[i] == k] for k in ks}
    zeros = np.zeros(n)

    def _em_at_theta(theta, max_em=200, em_tol=1e-10):
        # inner EM to convergence at FIXED theta; returns full marginal
        # loglik (Balan & Putter 2019, eq. 3) and the fitted pieces
        a = 1.0 / theta
        w = {k: 1.0 for k in ks}
        fit = None
        for _ in range(max_em):
            offs = np.asarray([math.log(w[cl[i]]) for i in range(n)])
            fit = cox_counting_process(zeros, t, e, Xa, offset=offs)
            eta = Xa @ fit["beta"]
            risk_w = [w[cl[i]] * math.exp(float(eta[i])) for i in range(n)]
            etimes = sorted(set(float(t[i]) for i in range(n) if e[i] == 1.0))
            dL = []
            for tk in etimes:
                d = sum(1 for i in range(n) if t[i] == tk and e[i] == 1.0)
                s0 = sum(risk_w[i] for i in range(n) if t[i] >= tk)
                dL.append(d / s0)
            H = []
            for i in range(n):
                hi = sum(dL[m] for m in range(len(etimes)) if etimes[m] <= t[i])
                H.append(hi * math.exp(float(eta[i])))
            D = [sum(1.0 for i in kidx[k] if e[i] == 1.0) for k in ks]
            L = [sum(H[i] for i in kidx[k]) for k in ks]
            w_new = {ks[j]: (D[j] + a) / (L[j] + a) for j in range(len(ks))}
            delta = max(abs(w_new[k] - w[k]) for k in ks)
            w = w_new
            if delta < em_tol:
                break
        # full marginal loglik: event terms + gamma Laplace-derivative term
        tidx = {tk: m for m, tk in enumerate(etimes)}
        ll = 0.0
        for i in range(n):
            if e[i] == 1.0:
                ll += float(eta[i]) + math.log(dL[tidx[float(t[i])]])
        for j in range(len(ks)):
            ll += (math.lgamma(a + D[j]) - math.lgamma(a)
                   + D[j] * math.log(theta)
                   - (a + D[j]) * math.log(1.0 + theta * L[j]))
        return ll, fit, w, etimes, dL, D, L

    # outer golden-section on log-theta maximizing the marginal loglik
    if theta is not None:
        theta = float(theta)
        ll_max, fit, w, etimes, dL, D, L = _em_at_theta(theta)
        beta = fit["beta"]
        cumL = []
        acc = 0.0
        for v in dL:
            acc += v
            cumL.append(acc)
        S_marg = [(1.0 + theta * v) ** (-1.0 / theta) for v in cumL]
        return RichResult(payload={
            "estimate": beta, "se": fit["se"], "theta": theta,
            "kendall_tau": theta / (theta + 2.0), "frailty": w,
            "baseline_times": np.asarray(etimes),
            "baseline_cumhaz": np.asarray(cumL),
            "marginal_survivor": np.asarray(S_marg),
            "loglik": ll_max, "n_outer": 0,
            "method": "Vaupel et al (1979) shared gamma frailty, fixed theta",
        })
    # NOTE: the marginal likelihood is flat at machine precision over a
    # ~1e-7 window around the argmax, so cross-language agreement on the
    # ESTIMATED theta is bounded at ~1e-6 (both arms sit inside the
    # plateau); the likelihood value itself agrees to <1e-12. Same
    # equally-optimal-answer class as the R/Python scan-order trap.
    lo, hi = math.log(1e-4), math.log(20.0)
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d2 = lo + gr * (hi - lo)
    fc = _em_at_theta(math.exp(c))[0]
    fd = _em_at_theta(math.exp(d2))[0]
    for _ in range(50):
        if fc > fd:
            hi, d2, fd = d2, c, fc
            c = hi - gr * (hi - lo)
            fc = _em_at_theta(math.exp(c))[0]
        else:
            lo, c, fc = c, d2, fd
            d2 = lo + gr * (hi - lo)
            fd = _em_at_theta(math.exp(d2))[0]
    theta = math.exp((lo + hi) / 2.0)
    ll_max, fit, w, etimes, dL, D, L = _em_at_theta(theta)
    beta = fit["beta"]
    outer = 50
    # marginal survivor at baseline covariates (x = 0)
    cumL = []
    acc = 0.0
    for v in dL:
        acc += v
        cumL.append(acc)
    S_marg = [(1.0 + theta * v) ** (-1.0 / theta) for v in cumL]
    return RichResult(payload={
        "estimate": beta,
        "se": fit["se"],
        "theta": theta,
        "kendall_tau": theta / (theta + 2.0),
        "frailty": w,
        "baseline_times": np.asarray(etimes),
        "baseline_cumhaz": np.asarray(cumL),
        "marginal_survivor": np.asarray(S_marg),
        "loglik": ll_max,
        "n_outer": outer,
        "method": "Vaupel et al (1979) shared gamma frailty, Klein (1992) EM, marginal survivor",
    })


shfrm = shared_frailty_marginal


def cheatsheet():
    return "shfrm(time, event, X, cluster) -> shared gamma frailty with marginal survivor output."
