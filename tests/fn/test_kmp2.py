"""Tests for kmp2.kamath_p_tuning_v2."""

from morie.fn import _array_core as np

from morie.fn.kmp2 import kamath_p_tuning_v2


def test_kmp2_basic():
    """Test basic functionality."""
    prefixes_by_layer = np.random.default_rng(42).normal(0, 1, 100)
    inputs_by_layer = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_p_tuning_v2(prefixes_by_layer, inputs_by_layer)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmp2_edge():
    """Test edge cases."""
    prefixes_by_layer = np.random.default_rng(42).normal(0, 1, 100)
    inputs_by_layer = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_p_tuning_v2(prefixes_by_layer, inputs_by_layer)
    assert isinstance(result, dict)
