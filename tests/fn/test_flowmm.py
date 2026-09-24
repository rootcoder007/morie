"""Tests for flowmm.max_flow_min_cut."""

import math

from morie.fn import _array_core as np
from morie.fn.flowmm import max_flow_min_cut


def test_flowmm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 5
    # Square non-negative capacity matrix
    G = rng.uniform(0.0, 10.0, (n, n))
    source = 0
    sink = n - 1
    result = max_flow_min_cut(G, source, sink)
    assert isinstance(result, dict)
    for key in ("estimate", "max_flow", "min_cut", "cut_size",
                "source_side", "augmentations", "n", "method"):
        assert key in result
    # Max-flow min-cut theorem: flow value equals min-cut capacity
    assert math.isclose(result["max_flow"], result["min_cut"])
    # Number of vertices matches input
    assert result["n"] == n
    # Max flow value is finite and non-negative
    assert math.isfinite(result["max_flow"])
    assert result["max_flow"] >= 0
    # Augmentations is a non-negative integer
    assert isinstance(result["augmentations"], int)
    assert result["augmentations"] >= 0
    # source_side contains the source but not the sink
    assert source in result["source_side"]
    assert sink not in result["source_side"]


def test_flowmm_edge():
    """Test edge case: graph with no path from source to sink."""
    n = 4
    # Disconnected graph: only a self-loop at the source, no path to sink
    G = [[0.0] * n for _ in range(n)]
    G[0][0] = 5.0
    source = 0
    sink = n - 1
    result = max_flow_min_cut(G, source, sink)
    assert isinstance(result, dict)
    # No augmenting path exists, so max flow and min cut are both 0
    assert result["max_flow"] == 0
    assert math.isclose(result["min_cut"], 0)
    # Number of vertices matches input
    assert result["n"] == n
    # Source side should contain only the source (no forward path)
    assert source in result["source_side"]
    assert sink not in result["source_side"]
