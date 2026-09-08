# SPDX-License-Identifier: AGPL-3.0-or-later
"""Seasonal Hybrid ESD anomaly detection (Twitter S-H-ESD)."""

import math

from . import _array_core as np

from ._richresult import RichResult
from ._sci_core import betainc
from .stlAn import stl_decompose

__all__ = ["ttsAn", "twitter_anomaly", "shesd", "t_quantile"]


def _t_cdf(t, v):
    # Student-t CDF via the regularised incomplete beta:
    # F(t) = 1 - I_x(v/2, 1/2)/2 for t >= 0, x = v/(v + t^2).
    x = v / (v + t * t)
    p = 0.5 * float(betainc(v / 2.0, 0.5, x))
    return 1.0 - p if t >= 0.0 else p


def t_quantile(p, v):
    """Student-t quantile by bisection on the betainc-based CDF
    (accurate to ~1e-13; anchored against R's qt in the tests)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p in (0,1) required")
    if p == 0.5:
        return 0.0
    neg = p < 0.5
    pp = 1.0 - p if neg else p
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, v) < pp:
        hi *= 2.0
        if hi > 1e300:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, v) < pp:
            lo = mid
        else:
            hi = mid
    q = 0.5 * (lo + hi)
    return -q if neg else q


def _median(v):
    s = sorted(v)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])


def _esd(res, k, alpha, hybrid=True):
    # Generalised ESD (Rosner 1983 as given by Hochenbaum et al. 2017,
    # eq (5)-(6)), with the S-H-ESD replacement of mean/sd by
    # median/MAD (their Sec. 3.5, eq (7)-(8), b = 1.4826).
    n = len(res)
    idx = list(range(n))
    vals = list(res)
    removed = []
    stats = []
    lams = []
    n_anom = 0
    for i in range(1, k + 1):
        if hybrid:
            ctr = _median(vals)
            scale = 1.4826 * _median([abs(v - ctr) for v in vals])
        else:
            ctr = sum(vals) / len(vals)
            scale = math.sqrt(sum((v - ctr) ** 2 for v in vals)
                              / (len(vals) - 1))
        if scale <= 0.0:
            break
        best = 0
        bdev = -1.0
        for j in range(len(vals)):
            d = abs(vals[j] - ctr)
            if d > bdev:
                bdev = d
                best = j
        C = bdev / scale
        # eq (6) with the original sample size n (Rosner 1983):
        # p = 1 - alpha / (2 (n - i + 1)), df = n - i - 1.
        p = 1.0 - alpha / (2.0 * (n - i + 1))
        tq = t_quantile(p, n - i - 1)
        lam = (n - i) * tq / math.sqrt((n - i - 1 + tq * tq) * (n - i + 1))
        stats.append(C)
        lams.append(lam)
        removed.append(idx[best])
        if C > lam:
            n_anom = i
        del vals[best]
        del idx[best]
        if len(vals) < 3:
            break
    return removed[:n_anom], stats, lams


def ttsAn(x, period, k=None, alpha=0.05, s_window=7, hybrid=True,
          direction="both"):
    """
    Seasonal Hybrid ESD (S-H-ESD) anomaly detection.

    Algorithm 1 of Hochenbaum, Vallis & Kejariwal (2017):
    1. Extract the seasonal component S_X with an STL decomposition
       (their STL Variant replaces the trend with the series median
       when forming the residual, Sec. 3.4.1).
    2. Residual R_X = X - S_X - X_tilde with X_tilde the series
       median (their eq (15)).
    3. Run generalised ESD (Rosner 1983; their eq (5)-(6)) on R_X for
       up to k anomalies: at step i the most extreme deviation
       C = max|x - centre| / scale is compared with the critical
       value lambda_i = (n-i) t_{p,n-i-1} /
       sqrt((n-i-1+t^2)(n-i+1)), p = 1 - alpha/(2(n-i+1)); the number
       of anomalies is the largest i with C_i > lambda_i.
       S-H-ESD uses the robust centre/scale: median and
       1.4826 * MAD (their Sec. 3.5, eqs (7)-(8)).

    Parameters
    ----------
    x : array-like
        Series (complete, at least two full cycles).
    period : int
        Observations per seasonal cycle.
    k : int, optional
        Maximum number of anomalies; default floor(0.49 n * ...) --
        their Algorithm 1 requires k <= n * 0.49; default
        max(1, floor(0.02 n)) mirroring the reference implementation
        default of 2 percent.
    alpha : float
        Significance level (their deployment uses 0.05, Sec. 4.2).
    s_window : int
        Seasonal loess span for the STL step.
    hybrid : bool
        True for S-H-ESD (median/MAD); False for S-ESD (mean/sd).
    direction : str
        "both", "pos" or "neg" -- one-sided detection keeps only
        residuals above/below the centre (their Sec. 4.3.1 one-tail
        perspectives).

    Returns
    -------
    result : RichResult
        Keys: anomalies (1-based positions), n_anomalies, statistics
        (C_i), critical_values (lambda_i), residual.

    References
    ----------
    Hochenbaum, J., Vallis, O. S. and Kejariwal, A. (2017),
    "Automatic anomaly detection in the cloud via statistical
    learning", arXiv:1704.07706. Algorithm 1, eqs (5)-(8), (14)-(15),
    Secs. 3.4-3.5. Rosner, B. (1983), "Percentage points for a
    generalized ESD many-outlier procedure", Technometrics 25(2),
    165-172 (ESD critical values). STL: Cleveland et al. (1990), JOS
    6(1), 3-73.
    Source PDF: /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
    hochenbaum-vallis-kejariwal-2017-twitter-shesd-anomaly-
    arxiv1704.07706.pdf
    """
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    xs = [float(v) for v in xv]
    n = len(xs)
    if k is None:
        k = max(1, int(math.floor(0.02 * n)))
    k = int(k)
    if k > int(0.49 * n):
        k = int(0.49 * n)
    if k < 1:
        raise ValueError("series too short for ESD")
    fit = stl_decompose(xs, period, s_window=s_window)
    S = fit["seasonal"]
    med = _median(xs)
    R = [xs[v] - S[v] - med for v in range(n)]
    if direction == "pos":
        Ruse = [max(r, 0.0) for r in R]
    elif direction == "neg":
        Ruse = [min(r, 0.0) for r in R]
    else:
        Ruse = R
    anoms, stats, lams = _esd(Ruse, k, alpha, hybrid=hybrid)
    anoms1 = sorted(a + 1 for a in anoms)
    return RichResult(payload={
        "anomalies": anoms1,
        "n_anomalies": len(anoms1),
        "statistics": stats,
        "critical_values": lams,
        "residual": R,
        "k": k,
        "alpha": alpha,
        "estimate": anoms1,
        "n": n,
        "method": "Seasonal Hybrid ESD (Hochenbaum-Vallis-Kejariwal 2017)",
    })


def twitter_anomaly(x, period, **kw):
    """Alias for ttsAn (original stub export name)."""
    return ttsAn(x, period, **kw)


shesd = ttsAn


def cheatsheet():
    return "ttsAn(x, period) -> S-H-ESD anomalies via STL seasonal removal + robust generalised ESD"

# public names resolved by fn/_lazy_map.json
twitteranomaly = ttsAn
