"""Tests for macohd.ma_cohens_d."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.macohd import ma_cohens_d


def test_macohd_basic():
    """Test basic functionality."""
    m1 = 5.0
    m2 = 5.5
    s1 = 1.0
    s2 = 1.2
    n1 = 30
    n2 = 35
    result = ma_cohens_d(m1, m2, s1, s2, n1, n2)
    assert isinstance(result, dict)
    expected_keys = {
        "d", "s_pooled", "var_d", "se_d", "j", "j_approx",
        "hedges_g", "var_g", "se_g", "df", "n", "method",
    }
    assert expected_keys.issubset(result.keys())
    assert result["df"] == n1 + n2 - 2
    assert result["n"] == n1 + n2
    assert math.isfinite(result["d"])
    assert math.isfinite(result["s_pooled"])
    assert math.isfinite(result["hedges_g"])
    assert math.isfinite(result["var_d"])
    assert math.isfinite(result["se_d"])
    assert result["s_pooled"] > 0.0


def test_macohd_edge():
    """Test edge cases."""
    # group size below 2 is documented as invalid
    with pytest.raises(ValueError):
        ma_cohens_d(5.0, 5.5, 1.0, 1.2, 1, 30)
    # negative standard deviation is documented as invalid
    with pytest.raises(ValueError):
        ma_cohens_d(5.0, 5.5, -1.0, 1.2, 30, 30)
    # zero pooled standard deviation is documented as invalid
    with pytest.raises(ValueError):
        ma_cohens_d(5.0, 5.5, 0.0, 0.0, 30, 30)
