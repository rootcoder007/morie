"""Tests for docl3.do_calculus_rules."""

from morie.fn import _array_core as np

from morie.fn.docl3 import do_calculus_rules


def test_docl3_basic():
    """Test basic functionality."""
    dag = {"A": [], "B": ["A"], "C": ["B"]}
    y = "C"
    z = "A"
    x = ()
    w = ()
    result = do_calculus_rules(dag, y, z, x, w)
    assert isinstance(result, dict)
    assert "rule1" in result
    assert "rule2" in result
    assert "rule3" in result
    assert "nrules" in result


def test_docl3_edge():
    """Test edge cases."""
    dag = {"A": [], "B": ["A"], "C": ["B"]}
    y = "C"
    z = "A"
    x = ()
    w = ()
    result = do_calculus_rules(dag, y, z, x, w)
    assert isinstance(result, dict)
    assert "rule1" in result
    assert "rule2" in result
    assert "rule3" in result
    assert "nrules" in result
