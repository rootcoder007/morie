"""Tests for variance_components_sigma2_u.variance_components_sigma2_u."""

from morie.fn import _array_core as np

from morie.fn.variance_components_sigma2_u import variance_components_sigma2_u


def test_ca7e6_basic():
    """Test basic functionality."""
    ms_between = 0.5
    ms_within = 0.5
    n_per_cluster = 0.5
    result = variance_components_sigma2_u(ms_between, ms_within, n_per_cluster)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca7e6_edge():
    """Test edge cases."""
    ms_between = 0.5
    ms_within = 0.5
    n_per_cluster = 0.5
    result = variance_components_sigma2_u(ms_between, ms_within, n_per_cluster)
    assert isinstance(result, dict)
