"""Tests for hmpre.geron_precision."""

import math

from morie.fn import _array_core as np

from morie.fn.hmpre import geron_precision


def test_hmpre_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    y_true = rng.integers(0, 2, 100)
    y_pred = rng.integers(0, 2, 100)
    result = geron_precision(y_true, y_pred)
    assert "precision" in result
    assert "tp" in result
    assert "fp" in result
    assert "fn" in result
    assert "f1" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    prec = float(result["precision"])
    assert math.isfinite(prec)
    assert 0.0 <= prec <= 1.0
    assert int(result["n"]) == 100
    assert math.isclose(prec, float(result["estimate"]))


def test_hmpre_edge():
    """Test edge cases."""
    # Perfect classifier: every prediction matches the truth.
    y_true = [0, 1, 1, 0, 1, 0, 1, 0]
    y_pred = [0, 1, 1, 0, 1, 0, 1, 0]
    result = geron_precision(y_true, y_pred)
    assert float(result["precision"]) == 1.0
    assert int(result["tp"]) == 4
    assert int(result["fp"]) == 0
    assert int(result["fn"]) == 0
    assert float(result["f1"]) == 1.0
    assert int(result["n"]) == 8


def test_every_printed_value_in_the_worked_example_reproduces():
    """Verify every printed value in the docstring's worked example reproduces."""
    r = geron_precision([1, 0, 1, 1, 0], [1, 1, 1, 0, 0])
    assert int(r["tp"]) == 2
    assert int(r["fp"]) == 1
    assert int(r["fn"]) == 1
    assert round(float(r["precision"]), 6) == 0.666667
    assert float(geron_precision([0, 1], [0, 1])["precision"]) == 1.0
