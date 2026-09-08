"""Tests for bkintr.burkov_ngram_interpolation."""

from morie.fn.bkintr import burkov_ngram_interpolation


def test_bkintr_basic():
    out = burkov_ngram_interpolation([0.8, 0.2], [0.75, 0.25])
    assert abs(out["estimate"] - 0.65) < 1e-12


def test_bkintr_edge():
    import pytest
    with pytest.raises(ValueError, match="sum to 1"):
        burkov_ngram_interpolation([0.5, 0.5], [0.5, 0.6])
