"""Tests for bsaclass.rangayyan_accuracy (eqs. 10.102-10.103)."""

from fractions import Fraction

import pytest

from morie.fn.bsaclass import rangayyan_accuracy


def test_rgacc_basic():
    """Raw (TP+TN)/N; sensitivity TP/(TP+FN), specificity TN/(TN+FP);
    weighted S+ P(A) + S- P(N); balanced = weighted at one half."""
    r = rangayyan_accuracy(tp=40, tn=30, fp=10, fn=20, prevalence=0.1)
    se, sp = 40 / 60, 30 / 40
    assert r["raw_accuracy"] == pytest.approx(70 / 100, abs=1e-15)
    assert r["sensitivity"] == pytest.approx(se, abs=1e-15)
    assert r["specificity"] == pytest.approx(sp, abs=1e-15)
    assert r["weighted_accuracy"] == pytest.approx(se * 0.1 + sp * 0.9, abs=1e-15)
    assert r["balanced_accuracy"] == pytest.approx((se + sp) / 2, abs=1e-15)
    assert r["accuracy"] == r["weighted_accuracy"]


def test_rgacc_edge():
    """Exact mode returns Fractions; raw accuracy is the weighted one at
    the test-set prevalence."""
    r = rangayyan_accuracy(tp=40, tn=30, fp=10, fn=20, exact=True)
    assert r["raw_accuracy"] == Fraction(7, 10)
    rw = rangayyan_accuracy(tp=40, tn=30, fp=10, fn=20, prevalence=Fraction(60, 100), exact=True)
    assert rw["weighted_accuracy"] == Fraction(7, 10)
