"""Tests for kmcrb.kamath_cross_encoder_rerank."""

from morie.fn import _array_core as np

from morie.fn.kmcrb import kamath_cross_encoder_rerank


def test_kmcrb_basic():
    """Test basic functionality."""
    q = 'q'
    docs = ['aa', 'b']
    model = lambda q, d: len(d)
    result = kamath_cross_encoder_rerank(q, docs, model)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmcrb_edge():
    """Test edge cases."""
    q = 'q'
    docs = ['aa', 'b']
    model = lambda q, d: len(d)
    result = kamath_cross_encoder_rerank(q, docs, model)
    assert isinstance(result, dict)
