"""Tests for morie.fn.mf1 — multi-class F1."""

from morie.fn.mf1 import mf1, multiclass_f1


def test_perfect_macro():
    r = multiclass_f1([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], average="macro")
    assert abs(r.estimate - 1.0) < 1e-10


def test_perfect_micro():
    r = multiclass_f1([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], average="micro")
    assert abs(r.estimate - 1.0) < 1e-10


def test_partial():
    r = multiclass_f1([0, 0, 1, 1], [0, 1, 0, 1], average="macro")
    assert 0 < r.estimate < 1


def test_alias():
    assert mf1 is multiclass_f1


def test_per_class_in_extra():
    r = multiclass_f1([0, 1, 0, 1], [0, 1, 0, 1], average="macro")
    assert "per_class" in r.extra
