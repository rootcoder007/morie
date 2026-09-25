"""Tests for wsmdir.wasserman_directed_graph."""

import math

import pytest

from morie.fn.wsmdir import wasserman_directed_graph

# Chain X -> Y: P(X=1) = 0.3, P(Y=1|X=0) = 0.2, P(Y=1|X=1) = 0.9.
_CHAIN = [
    {"parents": [], "cpt": {(): 0.3}},
    {"parents": [0], "cpt": {(0,): 0.2, (1,): 0.9}},
]


def test_wsmdir_basic():
    """Every factor and the joint follow straight from the CPTs."""
    out = wasserman_directed_graph(_CHAIN, [1, 1])
    assert out["n_nodes"] == 2
    assert out["factors"] == [pytest.approx(0.3), pytest.approx(0.9)]
    assert out["estimate"] == pytest.approx(0.3 * 0.9, rel=1e-12)
    assert out["log_joint"] == pytest.approx(math.log(0.3 * 0.9), rel=1e-12)

    out00 = wasserman_directed_graph(_CHAIN, [0, 0])
    assert out00["factors"] == [pytest.approx(0.7), pytest.approx(0.8)]
    assert out00["estimate"] == pytest.approx(0.7 * 0.8, rel=1e-12)

    # The four configurations of a two-node DAG form a distribution.
    total = sum(
        wasserman_directed_graph(_CHAIN, [a, b])["estimate"]
        for a in (0, 1)
        for b in (0, 1)
    )
    assert total == pytest.approx(1.0, rel=1e-12)


def test_wsmdir_three_node_collider():
    """A collider Z <- X, Y factorises as P(X) P(Y) P(Z | X, Y)."""
    px, py = 0.4, 0.25
    cpt = {(0, 0): 0.05, (0, 1): 0.5, (1, 0): 0.6, (1, 1): 0.95}
    dag = [
        {"parents": [], "cpt": {(): px}},
        {"parents": [], "cpt": {(): py}},
        {"parents": [0, 1], "cpt": cpt},
    ]

    for xv in (0, 1):
        for yv in (0, 1):
            for zv in (0, 1):
                out = wasserman_directed_graph(dag, [xv, yv, zv])
                p1 = cpt[(xv, yv)]
                expected = (
                    (px if xv else 1 - px)
                    * (py if yv else 1 - py)
                    * (p1 if zv else 1 - p1)
                )
                assert out["n_nodes"] == 3
                assert out["estimate"] == pytest.approx(expected, rel=1e-12)
                assert out["log_joint"] == pytest.approx(
                    math.log(expected), rel=1e-12
                )

    total = sum(
        wasserman_directed_graph(dag, [a, b, c])["estimate"]
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    )
    assert total == pytest.approx(1.0, rel=1e-12)

    # Marginalising Z out of the collider leaves P(X) P(Y).
    marg = sum(
        wasserman_directed_graph(dag, [1, 0, c])["estimate"] for c in (0, 1)
    )
    assert marg == pytest.approx(px * (1 - py), rel=1e-12)


def test_wsmdir_zero_probability_configuration():
    """A deterministic CPT drives the joint, and its log, to zero."""
    dag = [
        {"parents": [], "cpt": {(): 0.5}},
        {"parents": [0], "cpt": {(0,): 0.0, (1,): 1.0}},
    ]
    out = wasserman_directed_graph(dag, [0, 1])
    assert out["factors"] == [pytest.approx(0.5), 0.0]
    assert out["estimate"] == 0.0
    assert out["log_joint"] == float("-inf")
    # The permitted configurations still sum to one.
    total = sum(
        wasserman_directed_graph(dag, [a, b])["estimate"]
        for a in (0, 1)
        for b in (0, 1)
    )
    assert total == pytest.approx(1.0, rel=1e-12)


def test_wsmdir_edge():
    """Length, binarity, ordering, CPT coverage and range are enforced."""
    with pytest.raises(ValueError):
        wasserman_directed_graph(_CHAIN, [1, 1, 1])
    with pytest.raises(ValueError):
        wasserman_directed_graph(_CHAIN, [1, 2])
    # A parent that is not earlier in the ordering: a cycle in disguise.
    with pytest.raises(ValueError):
        wasserman_directed_graph(
            [
                {"parents": [1], "cpt": {(0,): 0.5, (1,): 0.5}},
                {"parents": [], "cpt": {(): 0.5}},
            ],
            [0, 0],
        )
    # CPT missing the parent configuration that this x selects.
    with pytest.raises(ValueError):
        wasserman_directed_graph(
            [
                {"parents": [], "cpt": {(): 0.3}},
                {"parents": [0], "cpt": {(0,): 0.2}},
            ],
            [1, 1],
        )
    # CPT entry outside [0, 1].
    with pytest.raises(ValueError):
        wasserman_directed_graph([{"parents": [], "cpt": {(): 1.4}}], [1])

    # A lone root node is just its own marginal.
    root = wasserman_directed_graph([{"parents": [], "cpt": {(): 0.3}}], [1])
    assert root["n_nodes"] == 1
    assert root["estimate"] == pytest.approx(0.3, rel=1e-12)
    assert root["factors"] == [pytest.approx(0.3)]


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmdir as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
