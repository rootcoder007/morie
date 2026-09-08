"""Tests for b105.burkov_lm_ch1_cosine_similarity."""

from morie.fn.b105 import burkov_lm_ch1_cosine_similarity


def test_b105_basic():
    assert abs(burkov_lm_ch1_cosine_similarity([1.0, 0.0], [0.0, 1.0])["estimate"]) == 0.0


def test_b105_edge():
    import pytest
    with pytest.raises(ValueError, match="zero vector"):
        burkov_lm_ch1_cosine_similarity([0.0], [1.0])
