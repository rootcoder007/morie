# morie.fn -- slice s05 (rootcoder007/morie)
"""Shewhart control chart: the three-sigma decision, and what it costs.

Shewhart, W. A. (1926), "Quality Control Charts", *Bell System
Technical Journal* 5(4), 593-603, doi:10.1002/j.1538-7305.1926.tb04213.x.
The paper was opened directly (Internet Archive scan of BSTJ 5:4) and
read as rendered page images: the four-step procedure of specification,
estimation, distribution and test is on p. 597; Figure 3, p. 600, is
the Inspection Engineering analysis sheet, whose final column computes
3*sigma for each monitored statistic -- 3 sigma_xbar = .0198010,
3 sigma_sigma = .0180106, 3 sigma_k = .0599001, 3 sigma_beta2 = .119800 --
and p. 601 describes the resulting dotted lines in Figure 4 as "the
limits within which the different statistics should lie, if the product
had been controlled".  The multiplier three is therefore taken from the
primary source, not from later practice.

CITATION LIMIT.  Shewhart's book-length treatment, *Economic Control of
Quality of Manufactured Product* (1931), where the choice of three is
argued at length rather than merely applied, is lending-restricted on
the Internet Archive and could not be opened; nothing is attributed
to it here.

The chart signals when |x_t - mu| > k sigma.  With k = 3 and a Gaussian
in-control distribution the false-alarm probability per point is
2(1 - Phi(3)) = 0.0026998, so the in-control average run length is
about 370 points: monitor a stable process daily for a year and expect
roughly one alarm that means nothing.  That number, not the 3, is what
makes the choice a trade-off, and it is returned alongside the
decisions so the alarm rate observed can be read against the alarm rate
expected.  A chart is a classifier, and its false-positive rate is a
property of the limit, not of the data.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["shewhart"]


def shewhart(x, mu, sigma, k=3.0):
    """Flag points outside k-sigma control limits.

    Parameters
    ----------
    x : array-like
        The monitored statistic, in time order.
    mu : float
        In-control centre line.
    sigma : float
        In-control standard deviation of the monitored statistic;
        must be strictly positive.
    k : float
        Limit multiplier.  Default 3, as in Shewhart (1926) Figure 3.

    Returns
    -------
    RichResult
        keys: ``estimate`` (number of signals), ``alerts`` (0/1 per
        point), ``z``, ``lcl``, ``ucl``, ``n_alerts``, ``alarm_rate``,
        ``false_alarm_prob``, ``arl0``, ``n``, ``k``, ``method``.

    References
    ----------
    Shewhart, W. A. (1926), *Bell System Technical Journal*
    5(4):593-603, doi:10.1002/j.1538-7305.1926.tb04213.x, Figure 3,
    p. 600.
    """
    xv = core.vec(x)
    n = len(xv)
    if n == 0:
        raise ValueError("shewhart: x is empty")
    m = float(mu)
    s = float(sigma)
    kk = float(k)
    if not (s > 0.0) or math.isinf(s):
        raise ValueError("shewhart: sigma must be a positive finite number")
    if math.isinf(m) or m != m:
        raise ValueError("shewhart: mu must be finite")
    if not (kk > 0.0) or math.isinf(kk):
        raise ValueError("shewhart: k must be a positive finite number")
    lcl = m - kk * s
    ucl = m + kk * s
    z = [(v - m) / s for v in xv]
    alerts = [1 if abs(zi) > kk else 0 for zi in z]
    n_alerts = sum(alerts)
    p = 2.0 * (1.0 - core.pnorm(kk))
    return RichResult(payload={
        "estimate": int(n_alerts), "alerts": alerts, "z": z,
        "lcl": lcl, "ucl": ucl, "n_alerts": int(n_alerts),
        "alarm_rate": n_alerts / n,
        "false_alarm_prob": p, "arl0": (1.0 / p) if p > 0.0 else float("inf"),
        "n": int(n), "k": kk,
        "method": "Shewhart (1926) k-sigma control chart, k = %g" % kk})


def cheatsheet():
    return ("shewh: three sigma buys ARL0 about 370 -- one meaningless alarm "
            "per 370 in-control points, which is the price of the limit")


# compact alias per ledger/NAMING.md
shewhartchart = shewhart
