"""Tests for hmlrh.geron_learning_rate_heuristic."""

import math

from morie.fn import _array_core as np

from morie.fn.hmlrh import geron_learning_rate_heuristic


def test_hmlrh_basic():
    """Test basic functionality with a diverging LR sweep."""
    n = 30
    # Geometrically increasing learning rates (strictly increasing, positive)
    lrs = [10.0 ** (-4 + 0.2 * i) for i in range(n)]
    # Losses decrease then blow up sharply at the tail
    losses = [max(0.1, 2.0 - 0.08 * i) for i in range(n - 5)]
    losses += [3.0, 20.0, 100.0, 500.0, 1000.0]
    lr_curve = list(zip(lrs, losses))
    result = geron_learning_rate_heuristic(lr_curve)
    assert isinstance(result, dict)
    for key in ("lr", "lr_diverge", "lr_min_loss", "min_loss",
                "diverged", "estimate", "n", "method"):
        assert key in result
    assert result["n"] == n
    # Loss blows up so the sweep must be flagged as diverged
    assert result["diverged"]
    for key in ("lr", "lr_diverge", "lr_min_loss", "min_loss"):
        assert math.isfinite(float(result[key]))
    # Per docstring: recommendation = lr_diverge / safety (safety=10 default)
    assert math.isclose(float(result["lr"]),
                        float(result["lr_diverge"]) / 10.0,
                        rel_tol=1e-12, abs_tol=1e-12)
    # Recommendation is strictly below the divergence point
    assert float(result["lr"]) < float(result["lr_diverge"])


def test_hmlrh_edge():
    """Test edge case with a non-diverging curve given as a mapping."""
    n = 10
    lrs = [10.0 ** (-4 + 0.2 * i) for i in range(n)]
    # Monotonically decreasing losses => running-min never overtaken by 4x
    losses = [2.0 - 0.1 * i for i in range(n)]
    lr_curve = {lr: loss for lr, loss in zip(lrs, losses)}
    result = geron_learning_rate_heuristic(lr_curve)
    assert isinstance(result, dict)
    for key in ("lr", "lr_diverge", "lr_min_loss", "min_loss",
                "diverged", "estimate", "n", "method"):
        assert key in result
    assert result["n"] == n
    # No spike => no divergence
    assert not result["diverged"]
    for key in ("lr", "lr_min_loss", "min_loss"):
        assert math.isfinite(float(result[key]))


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmlrh as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
