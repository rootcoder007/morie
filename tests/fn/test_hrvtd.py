"""Tests for hrvtd — HRV time-domain metrics."""

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.hrvtd import hrv_time_domain


def test_hrvtd_basic(rng):
    rr = 800 + rng.standard_normal(100) * 50
    result = hrv_time_domain(rr)
    assert isinstance(result, DescriptiveResult)
    assert "sdnn" in result.extra
    assert "rmssd" in result.extra
    assert "pnn50" in result.extra


def test_hrvtd_known_sdnn():
    rr = np.array([800.0, 850.0, 750.0, 800.0, 900.0])
    result = hrv_time_domain(rr)
    assert result.extra["sdnn"] > 0
    assert 0 <= result.extra["pnn50"] <= 100


def test_time_domain_hrv_recomputed():
    import math

    import pytest

    rr = [800.0, 860.0, 790.0, 805.0, 900.0, 780.0]
    m = sum(rr) / 6
    d = [rr[i + 1] - rr[i] for i in range(5)]
    r = hrv_time_domain(rr)
    assert r.extra["sdnn"] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in rr) / 5), rel=1e-13)
    assert r.extra["rmssd"] == pytest.approx(math.sqrt(sum(v * v for v in d) / 5), rel=1e-13)
    assert r.extra["pnn50"] == pytest.approx(100 * sum(abs(v) > 50 for v in d) / 5, rel=1e-14)
    assert r.extra["mean_hr"] == pytest.approx(60000 / m, rel=1e-14)
