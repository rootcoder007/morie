"""Tests for gh_c10_7.ghosal_frs_density."""

from morie.fn import _array_core as np

from morie.fn.gh_c10_7 import ghosal_frs_density


def test_gh_c10_7_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_frs_density(x)
    assert "grid" in result
    assert "density" in result
    assert np.all(np.isfinite(np.asarray(result["density"], dtype=float)))
    # Independent recomputation of the documented rate: n^(-s/(2s+1))
    nn = int(np.asarray(x, dtype=float).ravel().size)
    sv = 1.0
    expected_rate = nn ** (-sv / (2.0 * sv + 1.0))
    assert float(result["rate"]) == expected_rate
    assert result["adaptive"] is True
    assert result["K_fixed"] is None
    assert result["n"] == nn
    assert abs(float(result["mass"]) - 1.0) < 1e-6


def test_gh_c10_7_edge():
    """Test edge cases."""
    import pytest
    with pytest.raises(ValueError):
        ghosal_frs_density(np.array([42.0]))
