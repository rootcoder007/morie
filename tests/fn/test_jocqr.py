"""Tests for jocqr.joseph_conformalized_quantile_regression."""

import math

from morie.fn.jocqr import joseph_conformalized_quantile_regression


def test_jocqr_basic():
    """Scores max(lo - y, y - hi); qhat is the ceil((n+1)(1-alpha))-th
    smallest; both ends widen by qhat (Romano, Patterson & Candes 2019)."""
    lo = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    hi = [v + 2.0 for v in lo]
    y = [0.5, 3.4, 1.2, 4.8, 4.1, 8.0, 5.5, 8.2, 11.3, 10.0]
    s = sorted(max(a - c, c - b) for a, b, c in zip(lo, hi, y))
    k = math.ceil(11 * 0.8)
    r = joseph_conformalized_quantile_regression(lo, hi, y, [1.0], [3.0], alpha=0.2)
    assert (r["k"], r["qhat"]) == (k, s[k - 1])
    assert (r["lower"], r["upper"]) == ([1.0 - s[k - 1]], [3.0 + s[k - 1]])


def test_jocqr_edge():
    """Too few calibration points for the coverage: the interval is
    unbounded, as the finite-sample guarantee requires."""
    r = joseph_conformalized_quantile_regression([0.0, 1.0], [1.0, 2.0], [0.5, 1.5], [0.0], [1.0], alpha=0.1)
    assert r["k"] == 3 and r["qhat"] == math.inf
    assert r["lower"] == [-math.inf] and r["upper"] == [math.inf]


