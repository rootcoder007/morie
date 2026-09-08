"""Tests for b202.burkov_lm_ch2_lm_next_token."""

from morie.fn.b202 import burkov_lm_ch2_lm_next_token


def test_b202_basic():
    assert burkov_lm_ch2_lm_next_token("b", ["a", "b", "a"])["estimate"] == 1.0


def test_b202_edge():
    import pytest
    with pytest.raises(ValueError, match="at least 2"):
        burkov_lm_ch2_lm_next_token("a", ["a"])
