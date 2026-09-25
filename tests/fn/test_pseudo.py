"""Tests for pseudo.path_specific_effect (Avin, Shpitser and Pearl 2005)."""

import pytest

from morie.fn.pseudo import path_specific_effect


# 0 -> 1 -> 2, 0 -> 2, 0 -> 3 -> 2
B = [[0.0, 0.5, 0.3, 0.4], [0.0, 0.0, 0.8, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.5, 0.0]]


def test_pseudo_basic():
    """Linear path rule: the effect along g is the sum over x -> y paths
    inside g of the products of edge coefficients."""
    r = path_specific_effect(B, 0, 2)
    assert r["total"] == pytest.approx(0.3 + 0.5 * 0.8 + 0.4 * -0.5, abs=1e-15)
    assert r["direct"] == 0.3
    assert r["indirect"] == pytest.approx(0.5 * 0.8 + 0.4 * -0.5, abs=1e-15)
    assert r["estimate"] == r["total"]
    g = path_specific_effect(B, 0, 2, edges=[(0, 1), (1, 2)])
    assert g["estimate"] == pytest.approx(0.4, abs=1e-15)
    assert g["n_edges_used"] == 2
    assert path_specific_effect(B, 0, 2, edges=[(0, 3), (1, 2)])["estimate"] == 0.0


def test_pseudo_edge():
    """Cyclic B, a non-square B and x == y raise."""
    with pytest.raises(ValueError):
        path_specific_effect([[0.0, 1.0], [1.0, 0.0]], 0, 1)
    with pytest.raises(ValueError):
        path_specific_effect([[0.0, 1.0]], 0, 1)
    with pytest.raises(ValueError):
        path_specific_effect(B, 2, 2)
