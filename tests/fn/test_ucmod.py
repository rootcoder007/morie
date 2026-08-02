"""Tests for ucmod.unobserved_components."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ucmod import unobserved_components


def _signal(seed=0, n=240, period=12):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    trend = 5.0 + 0.05 * t
    seasonal = 3.0 * np.sin(2 * np.pi * t / period)
    return trend, seasonal, trend + seasonal + 0.3 * rng.standard_normal(n)


def test_ucmod_extracts_trend_and_seasonal():
    """Harvey (1989) structural decomposition: the extracted trend must
    track the true line and the seasonal must correlate strongly with
    the true cycle."""
    trend, seasonal, y = _signal()
    r = unobserved_components(y, period=12)
    fit_t = np.asarray(r["trend"], dtype=float)
    fit_s = np.asarray(r["seasonal"], dtype=float)
    assert np.corrcoef(fit_t, trend)[0, 1] > 0.95  # measured 0.965
    assert np.corrcoef(fit_s, seasonal)[0, 1] > 0.95
    # The pieces add back to something close to the data: residual sd
    # measured 0.88 against a data sd of ~2.5 (noise sd 0.3 plus what the
    # smoother leaves in the irregular component).
    resid = y - fit_t - fit_s
    assert float(np.std(resid)) < 1.0
    assert float(np.std(resid)) < 0.5 * float(np.std(y))


def test_ucmod_seasonal_component_sums_to_near_zero_over_a_period():
    _, _, y = _signal(seed=1)
    r = unobserved_components(y, period=12)
    s = np.asarray(r["seasonal"], dtype=float)
    # Rolling one-period sums stay near zero -- the defining constraint.
    sums = np.convolve(s, np.ones(12), mode="valid")
    assert float(np.abs(sums).mean()) < 0.5


def test_ucmod_rejects_too_short_series():
    with pytest.raises(ValueError, match="at least"):
        unobserved_components(np.arange(5.0), period=12)
