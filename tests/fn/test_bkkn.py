"""Tests for bkkn.burkov_kneser_ney."""

from morie.fn.bkkn import burkov_kneser_ney


def test_bkkn_basic():
    out = burkov_kneser_ney(2, 4, (2, 3, 10), d=0.75)
    assert abs(out["estimate"] - 0.425) < 1e-12


def test_bkkn_edge():
    import pytest
    with pytest.raises(ValueError, match="discount"):
        burkov_kneser_ney(2, 4, (2, 3, 10), d=1.5)
