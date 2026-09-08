"""Tests for krust."""

from morie.fn import _array_core as np
import pytest

from morie.fn.krust import kruskal_stress


def test_krust_basic():
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert kruskal_stress(D, D)["stress"] == pytest.approx(0.0)
    Dc = np.array([[0.0, 2.0], [2.0, 0.0]])
    assert kruskal_stress(D, Dc)["stress"] == pytest.approx(0.5)


def test_krust_edge():
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        kruskal_stress(D, np.zeros((2, 2)))  # zero distances
    with pytest.raises(ValueError):
        kruskal_stress(D, np.zeros((3, 3)))  # shape mismatch
