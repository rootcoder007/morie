"""Tests for morie.fn.graphspec."""

import math

import pytest

from morie.fn import graphspec as G

TRI2 = [
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0],
]


def test_estrada_and_communicability():
    assert G.estrada_index([[0, 1], [1, 0]]) == pytest.approx(2 * math.cosh(1))
    c = G.communicability([[0, 1], [1, 0]])
    assert c.matrix[0][0] == pytest.approx(math.cosh(1)) and c.matrix[0][1] == pytest.approx(math.sinh(1))
    assert c.estrada_index == pytest.approx(G.estrada_index([[0, 1], [1, 0]]))


def test_modularity_perron_randic():
    m = G.modularity_matrix(TRI2, [0, 0, 0, 1, 1, 1])
    assert m.modularity == pytest.approx(2 * (3 / 7 - 0.25))
    assert sorted(set(m.bisection[:3])) == [m.bisection[0]] and m.bisection[0] != m.bisection[5]
    pf = G.perron_frobenius([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    assert pf.eigenvalue == pytest.approx(2.0) and pf.ratio == pytest.approx(2.0)
    assert G.randic_index([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) == pytest.approx(math.sqrt(2))
    with pytest.raises(ValueError):
        G.perron_frobenius([[0, -1], [-1, 0]])


def test_label_propagation():
    two = [
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ]
    r = G.label_propagation(two, seed=3)
    assert r.membership == [0, 0, 0, 1, 1, 1] and r.modularity == pytest.approx(0.5)
