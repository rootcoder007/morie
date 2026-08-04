# morie.fn -- function file (rootcoder007/morie)
"""Bayesian online changepoint detection (Gaussian run model)."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['bocpd']


def bocpd(x, hazard=0.004, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
    """Bayesian online changepoint detection (Gaussian run model).

    A run-length posterior is propagated forward one datum at a time. The changepoint prior puts mass on exactly two outcomes -- the run grows or it resets to zero -- which is what makes the message passing linear in the number of run lengths. The conjugate-exponential model here is Normal-Inverse-Gamma over an unknown mean and variance, so the per-run predictive is a Student t. Nothing is truncated: the full run-length trellis is kept, so the result does not depend on a pruning threshold. A note on what is reported: P(r_t = 0 | x_1:t) is identically the hazard, because the changepoint branch and the growth branch are scaled by H and 1-H before the same normalisation, so the data cancel out of that entry. It carries no evidence and is returned as ``reset_prob`` only for completeness. ``cp_prob`` is P(r_t = 1 | x_1:t), the mass on a run that began one step ago, which is the entry that actually spikes at a changepoint.


    Formula: P(r_t=r_{t-1}+1, x_1:t) = P(r_{t-1},x_1:t-1) pi_t^(r) (1-H); P(r_t=0, x_1:t) = sum_r P(r_{t-1},x_1:t-1) pi_t^(r) H

    Parameters
    ----------
    x : array-like
        Observed univariate series.
    hazard : float
        Constant hazard H = 1/lambda of the geometric run-length prior.
    mu0 : float
        Prior mean.
    kappa0 : float
        Prior mean precision (pseudo-count).
    alpha0 : float
        Prior shape of the inverse-gamma variance.
    beta0 : float
        Prior scale of the inverse-gamma variance.

    Returns
    -------
    RichResult
        ``cp_prob`` (P(r_t = 1) at each t), ``reset_prob``, ``run_length`` (posterior mode), ``max_cp_prob``, ``hazard``, ``n``.

    References
    ----------
    Adams and MacKay (2007), Bayesian Online Changepoint Detection,
    arXiv:0710.3742.  Equations (2)-(5) for the recursion and the
    changepoint prior, Section 2.3 and Algorithm 1 for the
    conjugate-exponential update of the run-specific sufficient
    statistics.  Verified against the paper.
    """
    x = C.vec(x)
    n = len(x)
    H = float(hazard)
    mu = [float(mu0)]; kap = [float(kappa0)]
    al = [float(alpha0)]; be = [float(beta0)]
    R = [1.0]
    cp_prob, run_len = [], []
    for t in range(n):
        xt = x[t]
        pi = []
        for r in range(len(R)):
            df = 2.0 * al[r]
            s2 = be[r] * (kap[r] + 1.0) / (al[r] * kap[r])
            s = math.sqrt(s2)
            z = (xt - mu[r]) / s
            lg = (math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0)
                  - 0.5 * math.log(df * math.pi) - math.log(s)
                  - (df + 1.0) / 2.0 * math.log(1.0 + z * z / df))
            pi.append(math.exp(lg))
        growth = [R[r] * pi[r] * (1.0 - H) for r in range(len(R))]
        cp = sum(R[r] * pi[r] * H for r in range(len(R)))
        newR = [cp] + growth
        ev = sum(newR)
        newR = [v / ev for v in newR]
        nmu = [float(mu0)] + [(kap[r] * mu[r] + xt) / (kap[r] + 1.0) for r in range(len(R))]
        nkap = [float(kappa0)] + [kap[r] + 1.0 for r in range(len(R))]
        nal = [float(alpha0)] + [al[r] + 0.5 for r in range(len(R))]
        nbe = [float(beta0)] + [be[r] + kap[r] * (xt - mu[r]) ** 2
                                / (2.0 * (kap[r] + 1.0)) for r in range(len(R))]
        R, mu, kap, al, be = newR, nmu, nkap, nal, nbe
        cp_prob.append(R[1] if len(R) > 1 else float("nan"))
        run_len.append(max(range(len(R)), key=lambda i: R[i]))
    return RichResult(payload={
        "cp_prob": cp_prob, "run_length": run_len,
        "max_cp_prob": max(v for v in cp_prob if v == v),
        "reset_prob": H, "hazard": H, "n": n,
        "method": "Bayesian online changepoint detection (Normal-Inverse-Gamma)"})



def cheatsheet():
    return "bocpd: Bayesian online changepoint detection (Gaussian run model)."
