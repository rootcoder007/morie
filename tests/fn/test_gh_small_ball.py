"""Tests for gh_small_ball.ghosal_small_ball_prob."""

from morie.fn import _array_core as np

from morie.fn.gh_small_ball import ghosal_small_ball_prob


def test_gh_small_ball_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_small_ball_prob(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_small_ball_edge():
    """Test edge cases."""
    result = ghosal_small_ball_prob(np.array([42.0]))
    assert result["n"] == 1
