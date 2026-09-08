"""Tests for gh_dp_kl_nbhd.ghosal_dp_kl_nbhd_mass."""

from morie.fn import _array_core as np

from morie.fn.gh_dp_kl_nbhd import ghosal_dp_kl_nbhd_mass


def test_gh_dp_kl_nbhd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_kl_nbhd_mass(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_dp_kl_nbhd_edge():
    """Test edge cases."""
    result = ghosal_dp_kl_nbhd_mass(np.array([42.0]))
    assert result["n"] == 1
