"""Tests for gh_c3_15.ghosal_partspec_pt."""

from morie.fn import _array_core as np

from morie.fn.gh_c3_15 import ghosal_partspec_pt


def test_gh_c3_15_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_partspec_pt(x)
    assert "estimate" in result
    assert "cell_mass" in result
    assert "specified_levels" in result
    assert "method" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result["cell_mass"], dtype=float)))
    assert tuple(result["specified_levels"]) == (2, 4)
    assert isinstance(result["method"], str)


def test_gh_c3_15_edge():
    """Test edge cases: single-element input still works."""
    result = ghosal_partspec_pt(np.array([42.0]))
    # function returns estimate, cell_mass, specified_levels, method
    assert "estimate" in result
    assert "cell_mass" in result
    assert "specified_levels" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result["cell_mass"], dtype=float)))
    assert tuple(result["specified_levels"]) == (2, 4)
