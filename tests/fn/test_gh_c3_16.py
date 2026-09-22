"""Tests for gh_c3_16.ghosal_evsplit_pt."""

from morie.fn import _array_core as np

from morie.fn.gh_c3_16 import ghosal_evsplit_pt


def test_gh_c3_16_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_evsplit_pt(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c3_16_edge():
    """Test edge cases."""
    x = np.array([42.0])
    data = np.array([0.3, 0.5, 0.7])
    result = ghosal_evsplit_pt(x, data=data, depth=4, a_scale=2.0)
    assert result["n"] == len(data)
    assert result["n"] == 3
    assert result["depth"] == 4
    assert "estimate" in result
    assert "method" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
