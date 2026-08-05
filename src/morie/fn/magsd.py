# morie.fn -- function file (rootcoder007/morie)
"""Glass's delta and its large-sample variance."""

import math

from ._richresult import RichResult

__all__ = ["ma_glass_delta"]


def ma_glass_delta(m1, m2, s_ctrl, n1, n2):
    """Standardise by the control group's spread, not the pooled spread.

    Pooling the two standard deviations assumes the treatment changed the
    mean and left the variance alone.  When the treatment also changes the
    spread -- which is the usual case for anything that helps some people
    and not others -- the pooled denominator is itself an effect of the
    treatment, and the standardised difference is no longer comparable
    across studies.  Glass's delta uses the control spread, which the
    treatment cannot have touched.

    Formula: ``Delta = (m1 - m2)/s_ctrl`` with
    ``Var(Delta) = (n1 + n2)/(n1 n2) + Delta^2/(2 (n2 - 1))``, ``n2``
    being the control group -- Glass, McGaw & Smith (1981), Chapter 5;
    the variance is Hedges & Olkin (1985) eq. (5.10).

    Parameters
    ----------
    m1, m2 : float
        Treatment and control means.
    s_ctrl : float
        Control-group standard deviation, strictly positive.
    n1, n2 : int
        Treatment and control sample sizes; ``n2 >= 2``.

    Returns
    -------
    RichResult
        ``delta``, ``var``, ``se``, ``ci_lo``, ``ci_hi``, ``n1``, ``n2``.

    References
    ----------
    Glass, G. V., McGaw, B. and Smith, M. L. (1981).  Meta-Analysis in
    Social Research.  Sage, Chapter 5.  Variance from Hedges, L. V. and
    Olkin, I. (1985), Statistical Methods for Meta-Analysis, Academic
    Press, eq. (5.10).
    """
    s = float(s_ctrl)
    a = float(n1)
    b = float(n2)
    if s <= 0.0:
        raise ValueError("the control standard deviation must be positive")
    if a < 1.0 or b < 2.0:
        raise ValueError("need n1 >= 1 and n2 >= 2")
    d = (float(m1) - float(m2)) / s
    v = (a + b) / (a * b) + d * d / (2.0 * (b - 1.0))
    se = math.sqrt(v)
    return RichResult(payload={
        "delta": d, "var": v, "se": se,
        "ci_lo": d - 1.959963984540054 * se,
        "ci_hi": d + 1.959963984540054 * se,
        "n1": a, "n2": b,
        "method": "Glass's delta"})


def cheatsheet():
    return "magsd: Glass's delta standardised by the control SD"


# compact alias per ledger/NAMING.md
maglassdelta = ma_glass_delta
