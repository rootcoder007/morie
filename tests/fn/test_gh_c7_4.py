"""Tests for gh_c7_4.ghosal_norm_mix_con."""

from morie.fn import _array_core as np

from morie.fn.gh_c7_4 import ghosal_norm_mix_con


def test_gh_c7_4_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_norm_mix_con(x)
    assert "grid" in result
    assert "density" in result
    assert "reference_density" in result
    assert "hellinger_to_reference" in result
    assert "consistency" in result
    assert "requires" in result
    assert "n" in result
    assert "method" in result
    assert np.all(np.isfinite(np.asarray(result["density"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result["reference_density"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result["grid"], dtype=float)))
    assert np.isfinite(float(result["hellinger_to_reference"]))
    assert 0.0 <= float(result["hellinger_to_reference"]) <= 1.0
    assert result["n"] == 5


def test_gh_c7_4_edge():
    """Test edge cases."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = ghosal_norm_mix_con(x)
    assert result["n"] == 6
    assert np.all(np.isfinite(np.asarray(result["density"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result["reference_density"], dtype=float)))
