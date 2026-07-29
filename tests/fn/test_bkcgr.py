"""Tests for bkcgr.burkov_computational_graph."""

from morie.fn.bkcgr import burkov_computational_graph


def test_bkcgr_basic():
    g = [{"name": "sq", "op": "mul", "args": ["x", "x"]},
         {"name": "out", "op": "add", "args": ["sq", "x"]}]
    out = burkov_computational_graph(g, {"x": 3.0})
    assert out["output"] == 12.0
    assert out["gradients"]["x"] == 7.0


def test_bkcgr_edge():
    import pytest
    with pytest.raises(ValueError, match="unknown op"):
        burkov_computational_graph(
            [{"name": "a", "op": "pow", "args": ["x", "x"]}], {"x": 1.0})
