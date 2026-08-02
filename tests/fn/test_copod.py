"""Tests for copod."""

from morie.fn import _array_core as np
import pytest

from morie.fn.copod import copod

def test_copod_basic():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(150, 3))
    X[0] = 9.0
    out = copod(X)
    assert int(np.argmax(out["scores"])) == 0


def test_copod_edge():
    with pytest.raises(ValueError):
        copod(np.array([[1.0, 2.0]]))  # too few samples
    with pytest.raises(ValueError):
        copod(np.full((10, 2), np.nan))
