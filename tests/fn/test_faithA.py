"""Tests for faithA.faithfulness_assumption."""

from morie.fn import _array_core as np
import math
import pytest

from morie.fn.faithA import faithfulness_assumption


def test_faithA_basic():
    """Test basic functionality."""
    dag = {"A": [], "B": ["A"], "C": ["B"]}
    result = faithfulness_assumption(dag, "A", "C")
    assert isinstance(result, dict)
    assert len(result) > 0


def test_faithA_edge():
    """Test edge cases."""
    dag = {"A": [], "B": ["A"], "C": ["B"]}
    result = faithfulness_assumption(dag, "A", "C", z={"B"})
    assert isinstance(result, dict)
    assert len(result) > 0
