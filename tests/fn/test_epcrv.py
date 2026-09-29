"""Tests for morie.fn.epcrv -- Epidemic curve analysis."""

import pytest

from morie.fn.epcrv import epidemic_curve_analysis


class TestEpidemicCurveAnalysis:
    def test_basic(self):
        inc = [1, 2, 4, 8, 16, 32, 20, 10, 5, 3, 1]
        res = epidemic_curve_analysis(inc)
        assert res.measure == "epidemic_curve_analysis"
        assert res.extra["peak_day"] == 5

    def test_growth_rate(self):
        inc = [1, 2, 4, 8, 16, 32, 20, 10, 5, 3, 1]
        res = epidemic_curve_analysis(inc)
        assert res.extra["growth_rate"] > 0
        assert res.extra["doubling_time"] > 0

    def test_too_short(self):
        with pytest.raises(ValueError):
            epidemic_curve_analysis([1, 2])


def test_growth_rate_and_doubling_time_recomputed():
    """Log-linear slope over the ascending limb, doubling time log 2 / slope."""
    import math

    inc = [2, 3, 5, 8, 13, 21, 18, 12, 7]
    lg = [math.log(v) for v in inc[:6]]
    xs = list(range(6))
    mx, my = sum(xs) / 6, sum(lg) / 6
    slope = sum((a - mx) * (b - my) for a, b in zip(xs, lg)) / sum((a - mx) ** 2 for a in xs)
    res = epidemic_curve_analysis(inc, window=3)
    assert res.extra["growth_rate"] == pytest.approx(slope, rel=1e-10)
    assert res.extra["doubling_time"] == pytest.approx(math.log(2) / slope, rel=1e-10)
    assert res.extra["moving_average"] == pytest.approx([sum(inc[i : i + 3]) / 3 for i in range(7)], rel=1e-13)
