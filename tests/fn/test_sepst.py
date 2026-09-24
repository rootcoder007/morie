"""Tests for sepst.separation_set."""

from morie.fn import _array_core as np

from morie.fn.sepst import separation_set


def test_sepst_basic():
    """Test basic functionality."""
    # DAG: chain 0->1->2->3 plus alternative path 0->4->3
    dag = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 3)]
    result = separation_set(dag, 0, 3, maxsize=3)
    assert isinstance(result, dict)
    assert "sepset" in result or "estimate" in result or "statistic" in result


def test_sepst_edge():
    """Test edge cases."""
    # Minimal DAG with a single edge
    dag = [(0, 1)]
    result = separation_set(dag, 0, 1, maxsize=1)
    assert isinstance(result, dict)
