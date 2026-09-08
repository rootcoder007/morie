"""Tests for bkrep.burkov_repetition_penalty."""

from morie.fn.bkrep import burkov_repetition_penalty


def test_bkrep_basic():
    out = burkov_repetition_penalty([2.0, -2.0, 1.0], [0, 1], 2.0)
    assert out["penalised"] == [1.0, -4.0, 1.0]


def test_bkrep_edge():
    import pytest
    with pytest.raises(ValueError, match="out of range"):
        burkov_repetition_penalty([1.0], [3])
