# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""Anderson-Darling goodness-of-fit test (normal or exponential, parameters estimated)."""

from __future__ import annotations

import math

from ._containers import TestResult

# Gibbons & Chakraborti (2010) Table 4.7.1, the two composite-hypothesis rows (as rmorie):
# points of the modified statistic A* at alpha = 0.01, 0.025, 0.05, 0.10, 0.15
_AD_ALPHA = (0.01, 0.025, 0.05, 0.10, 0.15)
_AD_CRIT = {"norm": (1.035, 0.873, 0.752, 0.631, 0.561), "expon": (1.959, 1.591, 1.321, 1.062, 0.916)}


def _log_ndtr(z: float) -> float:
    """log Phi(z), accurate in the far lower tail."""
    if z > -30:
        return math.log(0.5 * math.erfc(-z / math.sqrt(2)))
    return -0.5 * z * z - math.log(-z * math.sqrt(2 * math.pi)) + math.log1p(-1 / (z * z))


def _p_from_crit(stat: float, crit, alpha):
    """Log-linear interpolation in the table; outside it the nearest bound, flagged."""
    pairs = sorted(zip(crit, alpha))
    if stat <= pairs[0][0]:
        return max(alpha), "upper"
    if stat >= pairs[-1][0]:
        return min(alpha), "lower"
    for (c0, a0), (c1, a1) in zip(pairs, pairs[1:]):
        if c0 <= stat <= c1:
            return math.exp(math.log(a0) + (math.log(a1) - math.log(a0)) * (stat - c0) / (c1 - c0)), None
    raise AssertionError("unreachable")  # pragma: no cover


def anderson_darling(x, dist: str = "norm") -> TestResult:
    r"""Anderson-Darling test of normality (or exponentiality) with estimated parameters.

    .. math::

        A^2 = -n - n^{-1}\sum (2i-1)[\ln F(z_{(i)}) + \ln(1 - F(z_{(n+1-i)}))]

    modified for the estimated parameters, :math:`A^* = A^2(1 + 0.75/n + 2.25/n^2)` (normal)
    or :math:`A^* = A^2(1 + 0.3/n)` (exponential), as rmorie's ``anderson_darling``.
    The normal p-value is D'Agostino & Stephens (1986, Table 4.9), as ``nortest::ad.test``;
    the exponential one interpolates Gibbons & Chakraborti (2010) Table 4.7.1.

    :param x: array-like of observations (non-finite values are dropped).
    :param dist: ``"norm"`` (default) or ``"expon"``.
    :return: TestResult with the modified statistic :math:`A^*` as ``statistic``; ``extra``
        carries the unmodified ``a_squared`` and ``p_bounded`` (``"upper"``/``"lower"`` when an
        exponential statistic falls outside the table).

    Examples
    --------
    >>> r = anderson_darling([5.1, 6.3, 4.8, 7.2, 5.9, 6.6, 5.4, 6.0, 5.5, 6.8])
    >>> round(r.p_value, 10)
    0.9693365958
    """
    if dist not in ("norm", "expon"):
        raise ValueError("dist must be 'norm' or 'expon'.")
    xs = sorted(float(v) for v in x if math.isfinite(float(v)))
    n = len(xs)
    if n < 3:
        raise ValueError("Anderson-Darling requires at least 3 finite observations.")
    mu = math.fsum(xs) / n
    if dist == "norm":
        s = math.sqrt(math.fsum((v - mu) ** 2 for v in xs) / (n - 1))
        if not math.isfinite(s) or s <= 0:
            raise ValueError("anderson_darling: 'x' has zero variance; the normal fit is degenerate.")
        z = [(v - mu) / s for v in xs]
        lf = [_log_ndtr(t) for t in z]
        lsf = [_log_ndtr(-t) for t in reversed(z)]
        mult = 1 + 0.75 / n + 2.25 / n**2
    else:
        if not math.isfinite(mu) or mu <= 0 or xs[0] < 0:
            raise ValueError('anderson_darling: dist = "expon" needs non-negative x with a positive mean.')
        z = [v / mu for v in xs]
        lf = [math.log(-math.expm1(-t)) if t > 0 else -math.inf for t in z]
        lsf = [-t for t in reversed(z)]
        mult = 1 + 0.3 / n
    a2 = -n - math.fsum((2 * i + 1) * (lf[i] + lsf[i]) for i in range(n)) / n
    astar = a2 * mult
    bounded = None
    if dist == "norm":
        aa = astar
        if aa < 0.2:
            p = 1 - math.exp(-13.436 + 101.14 * aa - 223.73 * aa**2)
        elif aa < 0.34:
            p = 1 - math.exp(-8.318 + 42.796 * aa - 59.938 * aa**2)
        elif aa < 0.6:
            p = math.exp(0.9177 - 4.279 * aa - 1.38 * aa**2)
        elif aa < 10:
            p = math.exp(1.2937 - 5.709 * aa + 0.0186 * aa**2)
        else:
            p = 3.7e-24
    else:
        p, bounded = _p_from_crit(astar, _AD_CRIT[dist], _AD_ALPHA)
    return TestResult(
        test_name="Anderson-Darling",
        statistic=astar,
        p_value=p,
        n=n,
        method=f"Anderson-Darling test ({dist})",
        extra={
            "a_squared": a2,
            "p_bounded": bounded,
            "critical_values": list(_AD_CRIT[dist]),
            "significance_levels": [100 * a for a in _AD_ALPHA],
        },
    )


ad = anderson_darling


def cheatsheet() -> str:
    return "anderson_darling({}) -> Anderson-Darling test for normality."
