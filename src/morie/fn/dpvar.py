# morie.fn -- function file (rootcoder007/morie)
"""Differentially private variance of bounded data.

Source opened: Karwa, V. and Vadhan, S. (2018).  Finite sample
differentially private confidence intervals.  ITCS 2018; arXiv:1711.03908,
Section 1.5 and Section 2.  The Laplace mechanism of Dwork et al. (2006),
quoted there as Lemma 2.2, releases f(x) + Lap(GS_f / eps) where GS_f is
the global sensitivity of f -- the largest |f(x) - f(x')| over datasets
differing in one row -- and the paper is explicit (page 5) that the
guarantee "relies on the data points being guaranteed to lie in the
range; otherwise, points need to be clamped to lie in the range".

The variance is released by composing two Laplace queries on data
clamped to [a, b], each spending half the budget, exactly as the paper's
Section 1.5 composes a private mean with a private second moment:

    mean_dp = mean(clamp(x)) + Lap(2 (b - a) / (n eps))
    m2_dp   = mean(clamp(x)^2) + Lap(2 (sup t^2 - inf t^2) / (n eps))
    var_dp  = m2_dp - mean_dp^2

with the suprema taken over t in [a, b], so inf t^2 is 0 when the
interval straddles zero and min(a^2, b^2) otherwise.  Sequential
composition makes the release eps-differentially private in total.

var_dp is clipped into [0, ((b - a)/2)^2]: the upper end is Popoviciu's
inequality, the sharp bound on the variance of a variable supported on
an interval, so post-processing to it costs no privacy and cannot
return a variance the data could not have had.

The noise is drawn deterministically by inverting the Laplace CDF at
distinct van der Corput points (bases 2 and 3), so the two language arms
land on identical numbers.  This is a reproducibility device, not a
privacy claim: a real deployment must use a cryptographic source.
"""

from __future__ import annotations

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["dp_variance"]


def _laplace(u, scale):
    """Inverse Laplace CDF at u in (0, 1), location zero."""
    d = u - 0.5
    s = 1.0 if d >= 0.0 else -1.0
    t = 1.0 - 2.0 * abs(d)
    if t < 1e-300:
        t = 1e-300
    return -scale * s * math.log(t)


def dp_variance(x, a, b, epsilon, seed=42):
    """eps-DP variance of data clamped to [a, b].

    Parameters
    ----------
    x : array-like
        Raw data.
    a, b : float
        Clamping bounds, a < b.
    epsilon : float
        Total privacy budget, strictly positive; split evenly between
        the mean query and the second-moment query.
    seed : int
        Index into the deterministic low-discrepancy stream.

    Returns
    -------
    result : dict
        Keys: estimate (private variance), var_dp, mean_dp, m2_dp,
        var_true, mean_true, sens_mean, sens_m2, scale_mean, scale_m2,
        n_clamped, epsilon, n.

    References
    ----------
    Karwa & Vadhan (2018), arXiv:1711.03908, Sections 1.5 and 2.
    Dwork, McSherry, Nissim & Smith (2006), TCC, LNCS 3876:265-284,
    doi:10.1007/11681878_14 (the Laplace mechanism).
    """
    xv = core.vec(x)
    n = len(xv)
    if n == 0:
        raise ValueError("empty input: x has no observations")
    a = float(a)
    b = float(b)
    if not (a < b):
        raise ValueError("bounds must satisfy a < b")
    eps = float(epsilon)
    if not (eps > 0.0):
        raise ValueError("epsilon must be strictly positive")
    si = int(seed)
    if si < 1:
        raise ValueError("seed must be at least 1")
    cl = []
    nclamp = 0
    for v in xv:
        w = a if v < a else (b if v > b else v)
        if w != v:
            nclamp += 1
        cl.append(w)
    mean_true = core.mean(cl)
    m2_true = core.mean([v * v for v in cl])
    hi2 = max(a * a, b * b)
    lo2 = 0.0 if (a <= 0.0 <= b) else min(a * a, b * b)
    sens_mean = (b - a) / n
    sens_m2 = (hi2 - lo2) / n
    half = eps / 2.0
    scale_mean = sens_mean / half
    scale_m2 = sens_m2 / half
    mean_dp = mean_true + _laplace(core.vdc(si, 2), scale_mean)
    m2_dp = m2_true + _laplace(core.vdc(si, 3), scale_m2)
    var_dp = m2_dp - mean_dp * mean_dp
    cap = ((b - a) / 2.0) ** 2
    if var_dp < 0.0:
        var_dp = 0.0
    elif var_dp > cap:
        var_dp = cap
    return RichResult(
        title="DP variance (bounded)",
        summary_lines=[("epsilon", eps), ("clamped", nclamp)],
        payload={
            "estimate": var_dp,
            "var_dp": var_dp,
            "mean_dp": mean_dp,
            "m2_dp": m2_dp,
            "var_true": m2_true - mean_true * mean_true,
            "mean_true": mean_true,
            "sens_mean": sens_mean,
            "sens_m2": sens_m2,
            "scale_mean": scale_mean,
            "scale_m2": scale_m2,
            "n_clamped": float(nclamp),
            "epsilon": eps,
            "n": n,
            "method": "DP variance (bounded)",
        },
    )


def cheatsheet():
    return "dpvar: DP variance (bounded)"


# compact alias per ledger/NAMING.md
dpvariance = dp_variance
