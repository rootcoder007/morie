"""Tests for doop.do_operator."""

from morie.fn import _array_core as np

from morie.fn.doop import do_operator


def _build_simple_dag():
    """Build a simple DAG: Z -> X -> Y, X -> Y, Z -> Y.

    Represented as an edge list (parent, child).
    """
    return [
        ("Z", "X"),
        ("X", "Y"),
        ("Z", "Y"),
    ]


def test_doop_basic():
    """Test basic functionality on a small graph.

    We intervene on X. Per the do-operator (graph surgery), every edge
    into X is deleted and every other structural equation is left
    untouched. The DAG has exactly one edge into X (Z -> X), and three
    edges total, so the surgery removes 1 edge and keeps 2. The graph has
    3 nodes.
    """
    dag = _build_simple_dag()

    # `x` documents the intervened variable. Use a simple identifier.
    result = do_operator(dag, "X")

    assert isinstance(result, dict)
    assert "nremoved" in result
    assert "nkept" in result
    assert "nnodes" in result

    # Independent computation of the expected edge counts.
    edges = list(dag)
    intervened = "X"
    nremoved = sum(1 for (u, v) in edges if v == intervened)
    nkept = sum(1 for (u, v) in edges if v != intervened)
    nnodes = len({n for edge in edges for n in edge})

    assert result["nremoved"] == nremoved
    assert result["nkept"] == nkept
    assert result["nnodes"] == nnodes


def test_doop_edge():
    """Test edge cases: intervening on a root node removes no edges."""
    dag = _build_simple_dag()

    # Z has no parents in this DAG, so do(Z = z) removes 0 edges and
    # keeps all 3.
    result = do_operator(dag, "Z")

    assert isinstance(result, dict)
    assert "nremoved" in result
    assert "nkept" in result
    assert "nnodes" in result

    edges = list(dag)
    intervened = "Z"
    nremoved = sum(1 for (u, v) in edges if v == intervened)
    nkept = sum(1 for (u, v) in edges if v != intervened)
    nnodes = len({n for edge in edges for n in edge})

    assert result["nremoved"] == nremoved
    assert result["nkept"] == nkept
    assert result["nnodes"] == nnodes
