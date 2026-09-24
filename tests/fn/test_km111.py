"""Tests for km111.kamath_ch7_faithfulness_metric."""

import pytest

from morie.fn import _array_core as np

from morie.fn.km111 import kamath_ch7_faithfulness_metric


def test_km111_basic():
    """Test basic functionality with a mix of 0/1 support indicators."""
    rng = np.random.default_rng(42)
    facts = rng.integers(0, 2, 20).tolist()
    result = kamath_ch7_faithfulness_metric(facts)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "n_supported" in result
    assert "n_facts" in result
    assert result["n_facts"] == 20
    assert result["n_supported"] == int(sum(facts))
    assert 0.0 <= result["estimate"] <= 1.0
    expected = sum(facts) / len(facts)
    assert abs(result["estimate"] - expected) < 1e-12


def test_km111_edge():
    """Test edge case: an empty fact list is explicitly invalid (0/0 undefined)."""
    with pytest.raises(ValueError):
        kamath_ch7_faithfulness_metric([])
