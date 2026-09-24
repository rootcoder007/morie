"""Tests for hmfth.geron_finetune_lm."""

import math

from morie.fn import _array_core as np

from morie.fn.hmfth import geron_finetune_lm


def _quadratic_task(th, batch):
    """A simple quadratic task L(theta) = (theta[0] - 3.0)**2."""
    th0 = th[0]
    loss = float((th0 - 3.0) ** 2)
    grad = np.array([2.0 * (th0 - 3.0)])
    return loss, grad


def test_hmfth_basic():
    """Test basic functionality."""
    model = _quadratic_task
    dataset = [1]
    epochs = 5
    lr = 0.1
    result = geron_finetune_lm(model, dataset, epochs=epochs, lr=lr, theta=[0.0])
    assert isinstance(result, dict)
    expected_keys = {
        "theta", "theta_init", "loss_history", "drift", "n_steps",
        "lr_schedule", "frozen", "grad_norms", "estimate", "n", "method",
    }
    assert expected_keys.issubset(set(result.keys()))


def test_hmfth_edge():
    """Test edge cases with freeze, batch_size, warmup, and weight_decay."""
    model = _quadratic_task
    dataset = [1, 2, 3]
    epochs = 3
    lr = 0.05
    result = geron_finetune_lm(
        model, dataset, epochs=epochs, lr=lr, theta=[1.0],
        freeze=[False], batch_size=2, warmup=2, weight_decay=0.01,
    )
    assert isinstance(result, dict)
    assert "theta" in result
    assert "loss_history" in result
    assert "drift" in result
    assert math.isfinite(result["drift"])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmfth as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
