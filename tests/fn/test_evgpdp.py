"""Tests for evgpdp.evt_gpd_pdf."""

from morie.fn import _array_core as np

from morie.fn.evgpdp import evt_gpd_pdf


def test_evgpdp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sigma = 1.0
    xi = 0.5
    result = evt_gpd_pdf(y, sigma, xi)
    assert isinstance(result, dict)
    assert "f" in result
    assert result["method"] == "GPD density (Coles 2001 eq. 4.2)"


def test_evgpdp_edge():
    """Test edge cases."""
    y = np.array([0.1, 0.2, 0.3])
    sigma = 1.0
    xi = 0.5
    result = evt_gpd_pdf(y, sigma, xi)
    assert isinstance(result, dict)
    assert "f" in result
    # Reference computation using the documented formula:
    # h(y) = (1/sigma) * (1 + xi*y/sigma)^(-1 - 1/xi)
    expected = (1.0 / sigma) * (1.0 + xi * y / sigma) ** (-1.0 - 1.0 / xi)
    assert all(abs(a - b) < 1e-9 for a, b in zip(result["f"], expected))
