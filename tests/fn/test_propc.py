"""Tests for propc.prophet_components."""

from morie.fn import _array_core as np
import pytest

from morie.fn.propc import prophet_components


def test_propc_recovers_slope_and_seasonality():
    """Linear trend + sinusoid: the slope estimate and the seasonal
    component are both known (Taylor & Letham 2018 decomposition)."""
    t = np.arange(144, dtype=float)
    seasonal = 1.5 * np.sin(2 * np.pi * t / 12)
    y = 2.0 + 0.1 * t + seasonal
    r = prophet_components(y, period=12)
    assert float(r["slope"]) == pytest.approx(0.1, abs=0.005)
    assert float(r["intercept"]) == pytest.approx(2.0, abs=0.3)
    fit_s = np.asarray(r["seasonal"], dtype=float)
    np.testing.assert_allclose(fit_s - fit_s.mean(), seasonal - seasonal.mean(), atol=0.1)


def test_propc_components_sum_to_the_fit():
    rng = np.random.default_rng(0)
    t = np.arange(96, dtype=float)
    y = 0.05 * t + np.sin(2 * np.pi * t / 12) + 0.2 * rng.standard_normal(96)
    r = prophet_components(y, period=12)
    total = np.asarray(r["trend"]) + np.asarray(r["seasonal"]) + np.asarray(r["residual"])
    np.testing.assert_allclose(total, y, atol=1e-8)


def test_propc_flat_series_has_flat_components():
    y = np.full(60, 3.0)
    r = prophet_components(y, period=12)
    assert abs(float(r["slope"])) < 1e-8
    assert float(np.abs(np.asarray(r["seasonal"], dtype=float)).max()) < 1e-6


def test_propc_rejects_too_short_series():
    with pytest.raises(ValueError, match=">="):
        prophet_components(np.arange(4.0), period=12)
