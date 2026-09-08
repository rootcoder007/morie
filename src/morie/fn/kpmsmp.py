# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Hall-Wellner simultaneous confidence band for a Kaplan-Meier curve.

Hall and Wellner (1980), "Confidence bands for a survival curve from
censored data", Biometrika 67(1):133-143, doi:10.1093/biomet/67.1.133.
Writing sigma^2(t) = sum_{t_j <= t} d_j / (n_j (n_j - d_j)) for the
Greenwood sum, the band over the whole observed range is

    S(t) +/- h_alpha * n^{-1/2} * (1 + n sigma^2(t)) * S(t),

where h_alpha is the upper alpha point of the supremum of a Brownian
bridge, i.e. the solution of the Kolmogorov equation

    P( sup |B(x)| > h ) = 2 sum_{k>=1} (-1)^{k+1} exp(-2 k^2 h^2) = alpha.

h is found here by bisection on that series rather than read from a
table, so any alpha may be used; at alpha = 0.05 it reproduces the
tabulated 1.3581.  Unlike the pointwise Greenwood interval the band
holds simultaneously over t, so it is strictly wider.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core  # noqa: F401

from ._richresult import RichResult
from .kpmnci import _risk_table

__all__ = ["km_simultaneous_band"]


def _sup_bb_tail(h):
    """P(sup |B| > h) for a Brownian bridge B."""
    tot = 0.0
    for k in range(1, 101):
        term = math.exp(-2.0 * k * k * h * h)
        tot += term if k % 2 == 1 else -term
        if term < 1e-18:
            break
    return 2.0 * tot


def _hall_wellner_crit(alpha):
    lo = 0.05
    hi = 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _sup_bb_tail(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def km_simultaneous_band(fit, alpha):
    """Hall-Wellner simultaneous band for S(t).

    Parameters
    ----------
    fit : mapping
        Risk table with keys ``time``, ``n_risk`` and ``n_event``.
    alpha : float
        Simultaneous error rate, e.g. 0.05.
    """
    t, nr, d, m = _risk_table(fit)
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("km_simultaneous_band: alpha must lie in (0, 1)")
    n = nr[0]
    h = _hall_wellner_crit(a)
    S = []
    sig2 = []
    s = 1.0
    v = 0.0
    for j in range(m):
        s *= 1.0 - d[j] / nr[j]
        if nr[j] > d[j]:
            v += d[j] / (nr[j] * (nr[j] - d[j]))
        else:
            v = float("inf")
        S.append(s)
        sig2.append(v)
    half = []
    lo = []
    hi = []
    for j in range(m):
        if sig2[j] == float("inf"):
            half.append(float("nan"))
            lo.append(float("nan"))
            hi.append(float("nan"))
            continue
        w = h / math.sqrt(n) * (1.0 + n * sig2[j]) * S[j]
        half.append(w)
        l = S[j] - w
        u = S[j] + w
        lo.append(0.0 if l < 0.0 else l)
        hi.append(1.0 if u > 1.0 else u)
    return RichResult(
        title="Hall-Wellner simultaneous band",
        summary_lines=[("times", m), ("h", h), ("alpha", a)],
        payload={
            "estimate": half[-1],
            "time": t,
            "surv": S,
            "half_width": half,
            "sigma2": sig2,
            "lower": lo,
            "upper": hi,
            "h": h,
            "alpha": a,
            "n_times": float(m),
            "n_risk_start": n,
            "n": m,
            "method": "S(t) +/- h_alpha n^-1/2 (1 + n sigma^2(t)) S(t), Hall & Wellner (1980)",
        },
    )


def cheatsheet():
    return "kpmsmp: Simultaneous confidence band for KM"


# compact alias per ledger/NAMING.md
kmsimultaneousband = km_simultaneous_band
