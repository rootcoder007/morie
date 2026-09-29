"""Tests for morie.fn.epdsg -- epidemic curve fitting."""

import pytest

from morie.fn import _array_core as np
from morie.fn.epdsg import epidemic_curve_fit


class TestEpiCurveFit:
    def test_lognormal_peak(self):
        from morie.fn._stats_core import lognorm

        t = np.arange(60, dtype=float)
        inc = 500 * lognorm.pdf(t, s=0.5, scale=np.exp(2.5))
        res = epidemic_curve_fit(inc, distribution="lognormal")
        assert 8 < res["peak_time"] < 20
        assert res["rmse"] < 5.0

    def test_gamma_fit(self):
        from morie.fn._stats_core import gamma

        t = np.arange(50, dtype=float)
        inc = 300 * gamma.pdf(t, a=5, scale=3)
        res = epidemic_curve_fit(inc, distribution="gamma")
        assert res["peak_time"] > 5
        assert res["distribution"] == "gamma"

    def test_total_cases(self):
        inc = np.array([1, 3, 8, 15, 20, 18, 10, 5, 2, 1], dtype=float)
        res = epidemic_curve_fit(inc)
        assert res["total_cases"] == pytest.approx(83.0)

    def test_short_array_raises(self):
        with pytest.raises(ValueError):
            epidemic_curve_fit(np.array([1.0, 2.0]))

    def test_invalid_dist_raises(self):
        with pytest.raises(ValueError):
            epidemic_curve_fit(np.ones(10), distribution="weibull")


def test_fitted_curve_is_the_scaled_density_at_the_reported_parameters():
    import math

    from morie.fn._stats_core import lognorm

    t = [float(i) for i in range(40)]
    inc = [300 * float(lognorm.pdf(v, s=0.6, scale=math.exp(2.2))) for v in t]
    res = epidemic_curve_fit(inc, distribution="lognormal")
    p = res["params"]
    want = [p["amplitude"] * float(lognorm.pdf(v, s=p["sigma"], scale=math.exp(p["mu"]))) for v in t]
    assert [float(v) for v in res["fitted_curve"]] == pytest.approx(want, rel=1e-12, abs=1e-14)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(inc, want)) / 40)
    assert res["rmse"] == pytest.approx(rmse, rel=1e-9, abs=1e-12)
    assert res["total_cases"] == pytest.approx(sum(inc), rel=1e-14)
