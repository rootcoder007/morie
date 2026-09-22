"""Tests for evgpdc.evt_gpd_cdf."""

from morie.fn import _array_core as np

from morie.fn.evgpdc import evt_gpd_cdf


def _gpd_cdf_formula(y, sigma, xi):
    """Independent implementation of the GPD CDF: H(y) = 1 - (1 + xi*y/sigma)^(-1/xi)."""
    return 1.0 - (1.0 + xi * y / sigma) ** (-1.0 / xi)


def test_evgpdc_basic():
    """Test basic functionality with scalars."""
    sigma = 2.0
    xi = 0.5
    y_vals = [0.0, 0.5, 1.0, 1.5]

    for v in y_vals:
        result = evt_gpd_cdf(v, sigma, xi)
        assert isinstance(result, dict)
        assert "F" in result
        assert "method" in result
        # Only a single scalar should come back (not a list).
        assert not isinstance(result["F"], list)
        expected = _gpd_cdf_formula(v, sigma, xi)
        assert abs(result["F"] - expected) < 1e-12


def test_evgpdc_array():
    """Test with an array of exceedances."""
    sigma = 2.0
    xi = 0.5
    y = np.array([0.0, 0.5, 1.0, 1.5])

    result = evt_gpd_cdf(y, sigma, xi)
    assert isinstance(result, dict)
    assert "F" in result
    assert "method" in result
    # When multiple values are passed, F should be a list of values.
    assert isinstance(result["F"], list)
    assert len(result["F"]) == len(y)

    expected = [_gpd_cdf_formula(float(v), sigma, xi) for v in y]
    for got, exp in zip(result["F"], expected):
        assert abs(got - exp) < 1e-12


def test_evgpdc_edge():
    """Test edge cases: y = 0 yields F = 0."""
    sigma = 3.0
    xi = 0.25
    y = 0.0

    result = evt_gpd_cdf(y, sigma, xi)
    assert isinstance(result, dict)
    assert "F" in result
    assert not isinstance(result["F"], list)
    # Per the formula, F(0) = 1 - 1 = 0.
    assert abs(result["F"] - 0.0) < 1e-12
