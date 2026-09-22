"""Tests for gh_dp_reg_post.ghosal_dp_regression_posterior."""

from morie.fn import _array_core as np

from morie.fn.gh_dp_reg_post import ghosal_dp_regression_posterior


def test_gh_dp_reg_post_basic():
    """Test basic functionality."""
    result = ghosal_dp_regression_posterior()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_gh_dp_reg_post_edge():
    """Test edge cases."""
    result = ghosal_dp_regression_posterior()
    assert isinstance(result, dict)
