"""Tests for hmovf.geron_overfitting."""

from morie.fn import _array_core as np

from morie.fn.hmovf import geron_overfitting


def test_hmovf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    train_err = rng.uniform(0, 1, 100)
    val_err = rng.uniform(0, 1, 100)
    result = geron_overfitting(train_err, val_err)
    assert isinstance(result, dict)
    assert "estimate" in result


def test_hmovf_edge():
    """Test edge cases."""
    # Use scalar inputs (small valid input)
    result = geron_overfitting(0.2, 0.3)
    assert isinstance(result, dict)
    assert "gap" in result


def test_every_printed_value_in_the_worked_example_reproduces():
    # First example from the docstring
    r = geron_overfitting(0.1, 0.5)
    assert float(r["gap"]) == 0.4
    assert float(r["ratio"]) == 5.0
    assert bool(r["overfitting"]) is True

    # Second example from the docstring
    c = geron_overfitting([1.0, 0.5, 0.2, 0.1], [1.0, 0.6, 0.5, 0.7])
    assert int(c["best_epoch"]) == 2
    assert float(c["best_val"]) == 0.5
    assert round(float(c["gap"]), 6) == 0.6
