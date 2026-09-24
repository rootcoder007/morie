"""Tests for vpc.variance_partition_coefficient."""

from morie.fn import _array_core as np

from morie.fn.vpc import variance_partition_coefficient


def test_vpc_basic():
    """Test basic functionality."""
    sigma2_u = 0.1
    result = variance_partition_coefficient(sigma2_u)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_vpc_edge():
    """Test edge cases."""
    sigma2_u = 0.1
    result = variance_partition_coefficient(sigma2_u)
    assert isinstance(result, dict)
