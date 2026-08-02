"""Tests for nbeat.nbeats_basis."""

from morie.fn import _array_core as np
import pytest

from morie.fn.nbeat import nbeats_basis


def _signal(n=144, period=12):
    t = np.arange(n, dtype=float)
    trend = 0.02 * t + 0.0005 * t**2
    seasonal = 2.0 * np.sin(2 * np.pi * t / period) + 1.0 * np.cos(4 * np.pi * t / period)
    return trend, seasonal, trend + seasonal


def test_nbeat_decomposes_trend_and_seasonality():
    """Noise-free polynomial + Fourier signal lies exactly in the basis, so
    the fit is essentially perfect and the components separate."""
    trend, seasonal, y = _signal()
    r = nbeats_basis(y, horizon=12, n_trend=3, n_season=5, period=12)
    assert float(r["r2"]) > 0.999
    fit_t = np.asarray(r["trend"], dtype=float)
    fit_s = np.asarray(r["seasonal"], dtype=float)
    # Components are identified up to a constant: compare after centring.
    np.testing.assert_allclose(fit_t - fit_t.mean(), trend - trend.mean(), atol=0.05)
    np.testing.assert_allclose(fit_s - fit_s.mean(), seasonal - seasonal.mean(), atol=0.05)


def test_nbeat_forecast_extends_the_pattern():
    _, _, y = _signal(n=132)
    h = 12
    r = nbeats_basis(y, horizon=h, n_trend=3, n_season=5, period=12)
    fc = np.asarray(r["forecast"], dtype=float)
    assert fc.shape == (h,)
    # The true continuation is known in closed form.
    t = np.arange(132, 132 + h, dtype=float)
    truth = 0.02 * t + 0.0005 * t**2 + 2.0 * np.sin(2 * np.pi * t / 12) + 1.0 * np.cos(4 * np.pi * t / 12)
    np.testing.assert_allclose(fc, truth, atol=0.3)


def test_nbeat_rejects_too_short_series():
    with pytest.raises(ValueError, match="at least"):
        nbeats_basis(np.arange(5.0), n_trend=3, n_season=5, period=12)
