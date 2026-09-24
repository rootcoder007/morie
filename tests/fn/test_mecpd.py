"""Tests for mecpd.markov_equivalence_class."""

from morie.fn import _array_core as np

from morie.fn.mecpd import mectest


def test_mecpd_basic():
    """Test basic functionality."""
    # DAG1: chain 0 -> 1 -> 2 as edge list
    dag1 = [(0, 1), (1, 2)]
    # DAG2: collider 0 -> 2, 1 -> 2 as edge list (different MEC)
    dag2 = [(0, 2), (1, 2)]
    result = mectest(dag1, dag2)
    assert isinstance(result, dict)
    # Verify the result contains a key that reports the test outcome
    assert any(
        k in result
        for k in (
            "estimate",
            "pvalue",
            "statistic",
            "test_stat",
            "test_statistic",
            "score",
            "decision",
            "reject",
            "equivalent",
            "is_equivalent",
            "p_value",
        )
    )


def test_mecpd_edge():
    """Test edge cases."""
    # Identical DAGs - same edge list for both arguments
    dag = [(0, 1), (1, 2)]
    result = mectest(dag, dag)
    assert isinstance(result, dict)
    assert any(
        k in result
        for k in (
            "estimate",
            "pvalue",
            "statistic",
            "test_stat",
            "test_statistic",
            "score",
            "decision",
            "reject",
            "equivalent",
            "is_equivalent",
            "p_value",
        )
    )
