"""Tests for gnnEx.gnn_explainer."""

import math

import pytest

from morie.fn.gnnEx import computation_graph, conditional_entropy


def test_gnnEx_basic():
    """The L-hop computation graph and H(Y) in nats."""
    adj = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    r = computation_graph(adj, 0, 2)
    assert r["nodes"] == [0, 1, 2] and r["edges"] == [(0, 1), (1, 2)]
    assert computation_graph(adj, 0, 3)["nodes"] == [0, 1, 2, 3]
    assert conditional_entropy([0.25, 0.75]) == pytest.approx(
        -(0.25 * math.log(0.25) + 0.75 * math.log(0.75)), rel=1e-15)


def test_gnnEx_edge():
    """A certain prediction carries no entropy."""
    assert conditional_entropy([1.0, 0.0]) == pytest.approx(0.0, abs=1e-15)


