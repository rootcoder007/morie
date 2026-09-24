"""Tests for hmhftn.geron_hf_trainer."""

import math

from morie.fn import _array_core as np

from morie.fn.hmhftn import geron_hf_trainer


def _make_lg():
    """Build a least-squares loss_and_grad function."""
    def lg(params, Xb, yb):
        r = Xb @ params - yb
        loss = float(np.mean(r ** 2))
        grad = (2.0 / len(yb)) * (Xb.T @ r)
        return loss, grad
    return lg


def test_hmhftn_basic():
    """Test basic functionality with both train and eval datasets."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    true_w = rng.normal(0, 1, p)
    y = X @ true_w + rng.normal(0, 0.1, n)

    model = {"params": np.zeros(p), "loss_and_grad": _make_lg()}
    args = {"epochs": 20, "batch_size": 8, "learning_rate": 0.05, "seed": 0}
    train_ds = (X, y)
    eval_ds = (X, y)

    result = geron_hf_trainer(model, args, train_ds, eval_ds)
    assert isinstance(result, dict)
    expected_keys = {"params", "best_params", "train_loss", "eval_loss",
                     "history", "best_epoch", "estimate", "n", "method"}
    assert expected_keys.issubset(set(result.keys()))
    assert math.isfinite(result["train_loss"])
    assert math.isfinite(result["eval_loss"])
    assert len(result["history"]) == 20
    assert result["n"] == n


def test_hmhftn_edge():
    """Test edge case without eval_ds (falls back to training loss)."""
    rng = np.random.default_rng(0)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    true_w = rng.normal(0, 1, p)
    y = X @ true_w + rng.normal(0, 0.1, n)

    model = {"params": np.zeros(p), "loss_and_grad": _make_lg()}
    args = {"epochs": 5, "batch_size": 4, "learning_rate": 0.1, "seed": 7}
    train_ds = (X, y)

    result = geron_hf_trainer(model, args, train_ds)
    assert isinstance(result, dict)
    assert "best_params" in result
    assert "history" in result
    assert len(result["history"]) == 5
    assert math.isfinite(result["train_loss"])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmhftn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
