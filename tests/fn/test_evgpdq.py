"""Tests for evgpdq.evt_gpd_quantile."""

from morie.fn import _array_core as np

from morie.fn.evgpdq import evt_gpd_quantile


def test_evgpdq_basic():
    """Test basic functionality."""
    p = np.asarray(0.5)
    sigma = 1.0
    xi = 0.1
    result = evt_gpd_quantile(p, sigma, xi)
    # Formula: y_p = (sigma/xi) * [(1 - p)^(-xi) - 1]
    expected = (sigma / xi) * ((1.0 - float(p)) ** (-xi) - 1.0)
    assert isinstance(result, dict)
    assert "y_p" in result
    assert "method" in result
    assert isinstance(result["y_p"], float)
    assert result["y_p"] == expected


def test_evgpdq_edge():
    """Test edge cases."""
    p = np.asarray([0.1, 0.5, 0.9])
    sigma = 2.0
    xi = 0.2
    result = evt_gpd_quantile(p, sigma, xi)
    assert isinstance(result, dict)
    assert "y_p" in result
    assert isinstance(result["y_p"], list)
    assert len(result["y_p"]) == 3
    for i, v in enumerate(p):
        expected_v = (sigma / xi) * ((1.0 - float(v)) ** (-xi) - 1.0)
        assert result["y_p"][i] == expected_v
