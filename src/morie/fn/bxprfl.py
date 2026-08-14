# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
r"""Baxter-King approximate band-pass filter.

Baxter, M. and King, R. G. (1999), "Measuring Business Cycles:
Approximate Band-Pass Filters for Economic Time Series", *The Review of
Economics and Statistics* 81(4), 575-593, doi:10.1162/003465399558454
(verified against Crossref).  The construction below was read from
rendered images of the NBER working-paper version (w5022), whose text
layer is unusable; the load-bearing passage is section 2.5,
"Constraints on specific points", printed page 8, equation (8).

The ideal band-pass filter has frequency response 1 on
w_low <= |w| <= w_high and 0 elsewhere.  Its Fourier coefficients follow
directly:

    b_0 = (w_high - w_low)/pi,
    b_h = (sin(h w_high) - sin(h w_low))/(pi h),   h != 0,

equivalently the difference of two low-pass filters, which is how the
paper derives them.  Truncating at lag K gives a finite moving average
but destroys the zero-frequency property that matters: page 8 requires
the band-pass weights to SUM TO ZERO, so that the filter annihilates a
constant and, applied to a series with a unit root, returns a stationary
one.  Equation (8) makes the adjustment additive,

    a_h = b_h + theta,   theta = (target - sum_{h=-K}^{K} b_h)/(2K + 1),

with target 1 for the low-pass case the paper writes out and target 0
for the band-pass case it states in the following paragraph.  Without
that adjustment the filter has non-zero gain at frequency zero and
leaks the trend straight into the "cycle" -- the single most common way
this filter is got wrong.

Because the filter is a two-sided moving average of half-width K, the
first and last K observations have no filtered value; they are returned
as NaN rather than silently padded, and ``n_valid`` reports the count.

Anchors, both exact: the weights sum to zero to machine precision, and
the filter therefore annihilates any constant series and any linear
trend (a symmetric zero-sum kernel kills both, since sum a_h = 0 and
sum h a_h = 0 by symmetry).  ``weight_sum`` reports the first.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["baxter_king"]


def bk_weights(p_low, p_high, K):
    """The 2K+1 constrained band-pass weights, index -K..K."""
    if not (p_low > 1.0):
        raise ValueError("baxter_king: p_low must exceed 1 period")
    if not (p_high > p_low):
        raise ValueError("baxter_king: p_high must exceed p_low")
    K = int(K)
    if K < 1:
        raise ValueError("baxter_king: K must be at least 1")
    w_high = 2.0 * math.pi / float(p_low)
    w_low = 2.0 * math.pi / float(p_high)
    b = [0.0] * (K + 1)
    b[0] = (w_high - w_low) / math.pi
    for h in range(1, K + 1):
        b[h] = (math.sin(h * w_high) - math.sin(h * w_low)) / (math.pi * h)
    tot = b[0] + 2.0 * sum(b[1:])
    theta = -tot / (2.0 * K + 1.0)
    return [b[abs(h)] + theta for h in range(-K, K + 1)]


def baxter_king(y, p_low=6.0, p_high=32.0, K=12):
    """Band-pass filter a series, passing cycles of length p_low..p_high.

    Parameters
    ----------
    y : array-like
        The series, in time order.
    p_low : float
        Shortest cycle length passed, in periods.  Must exceed 1.
    p_high : float
        Longest cycle length passed, in periods.
    K : int
        Truncation half-width; the filter is a 2K+1 moving average and
        loses K observations at each end.

    Returns
    -------
    RichResult
        ``cycle`` (filtered series, NaN in the first and last K slots),
        ``weights`` (the 2K+1 constrained weights), ``weight_sum``,
        ``n_valid``, ``estimate`` (mean of the valid filtered values),
        ``sd_cycle``, ``K``, ``n``.
    """
    yy = core.vec(y)
    n = len(yy)
    K = int(K)
    if n < 2 * K + 1:
        raise ValueError("baxter_king: series shorter than the 2K+1 filter window")
    a = bk_weights(p_low, p_high, K)
    out = [float("nan")] * n
    for t in range(K, n - K):
        s = 0.0
        for h in range(-K, K + 1):
            s += a[h + K] * yy[t - h]
        out[t] = s
    val = [u for u in out if u == u]
    return RichResult(
        title="Baxter-King band-pass filter",
        summary_lines=[("n", n), ("K", K), ("valid", len(val))],
        payload={
            "cycle": out,
            "weights": a,
            "weight_sum": sum(a),
            "n_valid": len(val),
            "estimate": core.mean(val) if val else float("nan"),
            "sd_cycle": core.sd(val, 1) if len(val) > 1 else float("nan"),
            "K": K,
            "n": n,
            "method": "Baxter and King (1999) Rev. Econ. Statist. 81(4):575-593, eq. (8) constraint",
        },
    )


def cheatsheet():
    return "bxprfl: ideal BP coefficients plus a constant so the weights sum to ZERO; else the trend leaks in"

# public names resolved by fn/_lazy_map.json
baxterking = bk_weights
