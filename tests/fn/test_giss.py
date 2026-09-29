"""Tests for giss.giss_anomaly."""

from morie.fn import _array_core as np
from morie.fn.giss import giss_anomaly


def test_giss_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = giss_anomaly(T, baseline)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_giss_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = giss_anomaly(T, baseline)
    assert isinstance(result, dict)


def test_base_period_anomaly_and_distance_weights_recomputed():
    """Anomalies against each station's 2000-2002 mean, stations weighted
    1 - d/1200, then the OLS trend of the combined series."""
    import pytest

    T = [[10.0, 10.5, 11.0, 11.8], [8.0, 8.2, 8.9, 9.5]]
    yrs = [2000.0, 2001.0, 2002.0, 2003.0]
    d = [300.0, 900.0]
    w = [1 - 300 / 1200, 1 - 900 / 1200]
    w = [v / sum(w) for v in w]
    base = [sum(r[:3]) / 3 for r in T]
    ser = [sum(w[i] * (T[i][j] - base[i]) for i in range(2)) for j in range(4)]
    mx, my = sum(yrs) / 4, sum(ser) / 4
    b = sum((a - mx) * (c - my) for a, c in zip(yrs, ser)) / sum((a - mx) ** 2 for a in yrs)
    r = giss_anomaly(T, years=yrs, base=(2000, 2002), dist=d)
    assert [float(v) for v in r["anomaly"]] == pytest.approx(ser, rel=1e-12, abs=1e-14)
    assert r["trend"] == pytest.approx(b, rel=1e-10)
    assert r["baseline"] == pytest.approx(sum(wi * bi for wi, bi in zip(w, base)), rel=1e-13)
