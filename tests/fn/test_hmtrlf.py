"""Tests for hmtrlf.geron_trl_finetune."""

from morie.fn import _array_core as np

from morie.fn.hmtrlf import geron_trl_finetune


def test_hmtrlf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # DPO: dataset items are (x_chosen, x_rejected) feature vectors
    pairs = [
        ([1.0, 0.0], [0.0, 1.0]),
        ([1.0, 1.0], [0.0, 1.0]),
    ]
    result = geron_trl_finetune(None, pairs, method="dpo", epochs=50, lr=0.1, beta=1.0)
    assert isinstance(result, dict)
    assert "loss" in result
    assert "loss_curve" in result
    assert "margin" in result
    assert "theta" in result
    assert "dpo" in str(result["method"]).lower()


def test_hmtrlf_edge():
    """Test edge cases."""
    # PPO: items are (x, logp_old, advantage)
    items = [([1.0], 0.0, 1.0)]
    result = geron_trl_finetune(None, items, method="ppo", epochs=10, lr=0.1, clip_eps=0.2)
    assert isinstance(result, dict)
    assert "loss" in result
    assert "clipped_fraction" in result
    assert "ppo" in str(result["method"]).lower()


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmtrlf as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
