"""Tests for dpkb.k_anonymity (Sweeney 2002; alias of kanon)."""

import pytest

from morie.fn.dpkb import k_anonymity


def test_dpkb_basic():
    """Equivalence classes are the distinct quasi-identifier rows; the
    release is k-anonymous when the smallest class has at least k
    records.  A 6-record matrix counts as 6 records, not 18 cells."""
    X = [[1.0, 2.0, 3.0]] * 6
    q = [[30, "M"], [30, "M"], [40, "F"], [40, "F"], [40, "F"], [50, "M"]]
    q = [[a, 1 if b == "M" else 0] for a, b in q]
    r = k_anonymity(X, q, 2)
    assert r["n"] == 6
    assert r["n_classes"] == 3
    assert r["min_class_size"] == 1 and r["max_class_size"] == 3
    assert r["mean_class_size"] == pytest.approx(2.0, abs=1e-15)
    assert r["satisfies"] in (0, 0.0, False)
    assert r["n_violating"] == 1


def test_dpkb_edge():
    """k = 1 is always satisfied; mismatched lengths and k < 1 raise."""
    X = [[0.0]] * 4
    q = [[1], [1], [2], [2]]
    assert k_anonymity(X, q, 2)["satisfies"] in (1, 1.0, True)
    with pytest.raises(ValueError):
        k_anonymity(X[:3], q, 2)
    with pytest.raises(ValueError):
        k_anonymity(X, q, 0)
