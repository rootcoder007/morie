"""Tests for fciag.fci_algorithm."""

from morie.fn import _array_core as np
import pytest

from morie.fn.fciag import fci_algorithm


def test_fciag_basic():
    rng = np.random.default_rng(42)
    n = 4000
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    cc = a + b + rng.normal(scale=0.4, size=n)
    d = cc + rng.normal(scale=0.4, size=n)
    out = fci_algorithm(np.column_stack([a, b, cc, d]), names=["A", "B", "C", "D"])
    edges = {frozenset(e) for e in out["edges"]}
    assert frozenset({"A", "C"}) in edges
    assert frozenset({"A", "B"}) not in edges  # marginally independent
    assert ("A", "C", "B") in out["colliders"] or ("B", "C", "A") in out["colliders"]


def test_fciag_edge():
    with pytest.raises(ValueError):
        fci_algorithm(np.zeros((10, 2)))  # < 3 variables
    with pytest.raises(ValueError):
        fci_algorithm(np.zeros((10, 3)), alpha=1.5)
