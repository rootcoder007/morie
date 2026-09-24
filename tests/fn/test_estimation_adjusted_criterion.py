"""Tests for estimation_adjusted_criterion.estimation_adjusted_criterion."""

import math

from morie.fn import _array_core as np

from morie.fn.estimation_adjusted_criterion import (
    estimation_adjusted_criterion,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e6_basic():
    """Test basic functionality."""
    akv = 1.5
    v_ok = 4.0
    vkv = 3.2
    res = estimation_adjusted_criterion(akv, v_ok, vkv)
    assert isinstance(res, dict)
    assert "value" in res
    assert "method" in res
    expected = akv + vkv / (2.0 * v_ok)
    assert math.isclose(res["value"], expected)
    assert math.isfinite(res["value"])


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e6_edge():
    """Test edge cases."""
    akv = 0.0
    v_ok = 1.0
    vkv = 0.0
    res = estimation_adjusted_criterion(akv, v_ok, vkv)
    assert isinstance(res, dict)
    assert "value" in res
    assert math.isfinite(res["value"])
    assert math.isclose(res["value"], 0.0)
