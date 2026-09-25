"""Tests for rgspe.rangayyan_specificity."""

import pytest

from morie.fn.bsaclass import rangayyan_specificity


def test_rgspe_basic():
    """Specificity = TN / (TN + FP); FPF = 1 - specificity."""
    r = rangayyan_specificity(35, 15)
    assert r["specificity"] == pytest.approx(0.7, rel=1e-15)
    assert r["fpf"] == pytest.approx(1 - r["specificity"], rel=1e-15)
    assert rangayyan_specificity([[45, 5], [15, 35]])["specificity"] == r["specificity"]


def test_rgspe_edge():
    with pytest.raises(ValueError, match="undefined"):
        rangayyan_specificity(0, 0)
    with pytest.raises(ValueError, match="negative"):
        rangayyan_specificity(-1, 2)


