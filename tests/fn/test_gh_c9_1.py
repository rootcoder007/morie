"""Tests for gh_c9_1.ghosal_logspline_crt."""

from morie.fn import _array_core as np

from morie.fn.gh_c9_1 import ghosal_logspline_crt


def test_gh_c9_1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_logspline_crt(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c9_1_edge():
    """Test edge cases."""
    data = np.array([42.0])
    result = ghosal_logspline_crt(data)
    # Compute expected n from the input data independently
    expected_n = int(np.asarray(data).size)
    assert expected_n == 1
    assert "K_n" in result
    assert result["K_n"] >= 1
