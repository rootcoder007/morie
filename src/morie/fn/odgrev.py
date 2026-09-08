# morie.fn -- function file (rootcoder007/morie)
"""Outbreak detection by online changepoint analysis of counts."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['outbrkdet', 'outbreak_detection']


def outbrkdet(counts, hazard=0.01, a0=1.0, b0=1.0):
    """Outbreak detection by online changepoint analysis of counts.

    Case counts are not Gaussian, so the Gaussian run model is replaced by the other standard conjugate-exponential pair: a Gamma prior on the Poisson rate, whose marginal predictive is negative binomial. The recursion, the two-outcome changepoint prior and the sufficient-statistic update are unchanged from Adams and MacKay's Algorithm 1; only the emission model differs. A note on what is reported: P(r_t = 0 | x_1:t) is identically the hazard, because the changepoint branch and the growth branch are scaled by H and 1-H before the same normalisation, so the data cancel out of that entry. It carries no evidence and is returned as ``reset_prob`` only for completeness. ``cp_prob`` is P(r_t = 1 | x_1:t), the mass on a run that began one step ago, which is the entry that actually spikes at a changepoint.


    Formula: same run-length recursion as bocpd with a Gamma-Poisson run model; predictive P(x|a,b) = Gamma(x+a)/(Gamma(a) x!) (b/(b+1))^a (1/(b+1))^x

    Parameters
    ----------
    counts : array-like
        Non-negative integer case counts per period.
    hazard : float
        Constant hazard of the geometric run-length prior.
    a0 : float
        Gamma prior shape on the Poisson rate.
    b0 : float
        Gamma prior rate on the Poisson rate.

    Returns
    -------
    RichResult
        ``cp_prob`` (P(r_t = 1)), ``reset_prob``, ``run_length``, ``max_cp_prob``, ``alarm`` (indices with cp_prob > 0.5), ``n``.

    References
    ----------
    Adams and MacKay (2007), Bayesian Online Changepoint Detection,
    arXiv:0710.3742.  Equations (2)-(5) for the recursion and the
    changepoint prior, Section 2.3 and Algorithm 1 for the
    conjugate-exponential update of the run-specific sufficient
    statistics.  Verified against the paper.
    """
    y = C.vec(counts)
    n = len(y)
    H = float(hazard)
    a = [float(a0)]; b = [float(b0)]
    R = [1.0]
    cp_prob, run_len = [], []
    for t in range(n):
        xt = y[t]
        if xt < 0:
            raise ValueError("counts must be non-negative")
        pi = []
        for r in range(len(R)):
            lg = (math.lgamma(xt + a[r]) - math.lgamma(a[r])
                  - math.lgamma(xt + 1.0)
                  + a[r] * math.log(b[r] / (b[r] + 1.0))
                  - xt * math.log(b[r] + 1.0))
            pi.append(math.exp(lg))
        growth = [R[r] * pi[r] * (1.0 - H) for r in range(len(R))]
        cp = sum(R[r] * pi[r] * H for r in range(len(R)))
        newR = [cp] + growth
        ev = sum(newR)
        newR = [v / ev for v in newR]
        a = [float(a0)] + [a[r] + xt for r in range(len(R))]
        b = [float(b0)] + [b[r] + 1.0 for r in range(len(R))]
        R = newR
        cp_prob.append(R[1] if len(R) > 1 else float("nan"))
        run_len.append(max(range(len(R)), key=lambda i: R[i]))
    return RichResult(payload={
        "cp_prob": cp_prob, "run_length": run_len,
        "max_cp_prob": max(v for v in cp_prob if v == v),
        "reset_prob": H,
        "alarm": [i for i, v in enumerate(cp_prob) if v == v and v > 0.5], "n": n,
        "method": "Outbreak detection (Gamma-Poisson online changepoint)"})


outbreak_detection = outbrkdet


def cheatsheet():
    return "odgrev: Outbreak detection by online changepoint analysis of counts."
