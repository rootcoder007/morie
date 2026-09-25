"""Tests for rgppv.rangayyan_ppv."""

import pytest

from morie.fn.bsaclass import rangayyan_ppv


def test_rgppv_basic():
    """PPV = TP / (TP + FP), from counts or the table [[TP, FN], [FP, TN]]."""
    r = rangayyan_ppv(45, 15)
    assert r["ppv"] == pytest.approx(0.75, rel=1e-15)
    assert rangayyan_ppv([[45, 5], [15, 35]])["ppv"] == r["ppv"]


def test_rgppv_edge():
    """Bayes at a screening prevalence: Se p / (Se p + (1 - Sp)(1 - p));
    a 0.9-sensitive, 0.7-specific test at 1 in 1000 has PPV about 0.003."""
    r = rangayyan_ppv(45, 15, prevalence=0.001, sensitivity=0.9, specificity=0.7)
    want = 0.9 * 0.001 / (0.9 * 0.001 + 0.3 * 0.999)
    assert r["ppv_at_prevalence"] == pytest.approx(want, rel=1e-15)
    with pytest.raises(ValueError, match="undefined"):
        rangayyan_ppv(0, 0)
    with pytest.raises(ValueError, match="both"):
        rangayyan_ppv(1, 1, prevalence=0.1)


