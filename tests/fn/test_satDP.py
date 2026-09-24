"""Tests for satDP.dpll."""

from morie.fn import _array_core as np

from morie.fn.satDP import dpll


def test_satDP_basic():
    """Test basic functionality."""
    # (x1 or x2) and (not x1 or x2) is satisfiable
    cnf = [[1, 2], [-1, 2]]
    result = dpll(cnf)
    assert isinstance(result, dict)
    assert "satisfiable" in result
    assert "n_vars" in result
    assert "n_clauses" in result
    assert "decisions" in result
    assert "propagations" in result
    assert "pure_literals" in result
    assert "method" in result
    assert "assignment" in result
    assert "model" in result
    assert result["n_vars"] == 2
    assert result["n_clauses"] == 2
    assert result["satisfiable"] is True
    # model should be a full assignment dict over 1..n_vars
    assert isinstance(result["model"], dict)
    assert len(result["model"]) == 2
    for k, v in result["model"].items():
        assert isinstance(k, int)
        assert 1 <= k <= 2
        assert isinstance(v, bool)
    # assignment should be a dict of variable -> bool
    assert isinstance(result["assignment"], dict)
    for v in result["assignment"].values():
        assert isinstance(v, bool)


def test_satDP_edge():
    """Test edge cases."""
    # x1 and not x1 is unsatisfiable
    cnf = [[1], [-1]]
    result = dpll(cnf)
    assert isinstance(result, dict)
    assert "satisfiable" in result
    assert "n_vars" in result
    assert "n_clauses" in result
    assert "decisions" in result
    assert "propagations" in result
    assert "pure_literals" in result
    assert "method" in result
    assert "assignment" in result
    assert "model" in result
    assert result["satisfiable"] is False
    assert result["n_vars"] == 1
    assert result["n_clauses"] == 2
    # model may be empty for unsatisfiable; just verify it's a dict
    assert isinstance(result["model"], dict)
    # assignment should be a dict of variable -> bool
    assert isinstance(result["assignment"], dict)
