"""Tests for hmdpo.geron_dpo."""

from morie.fn import _array_core as np

from morie.fn.hmdpo import geron_dpo


def test_hmdpo_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    B = 50
    # log-probabilities must be <= 0
    pi = rng.uniform(-10.0, -0.1, (B, 2))
    pi_ref = rng.uniform(-10.0, -0.1, (B, 2))
    preferences = rng.integers(0, 2, B)
    beta = 0.8
    result = geron_dpo(pi, pi_ref, preferences, beta)
    assert isinstance(result, dict)
    expected_keys = {
        "loss", "per_pair_loss", "margin", "reward_chosen",
        "reward_rejected", "accuracy", "prob_preferred",
        "estimate", "n", "method",
    }
    for key in expected_keys:
        assert key in result
    import math
    assert math.isfinite(result["loss"])
    assert result["n"] == B
    assert 0.0 <= result["accuracy"] <= 1.0
    assert len(result["margin"]) == B
    assert len(result["prob_preferred"]) == B
    assert len(result["per_pair_loss"]) == B


def test_hmdpo_edge():
    """Test edge cases: default preferences (all zeros), small batch."""
    rng = np.random.default_rng(7)
    B = 20
    pi = rng.uniform(-10.0, -0.1, (B, 2))
    pi_ref = rng.uniform(-10.0, -0.1, (B, 2))
    beta = 0.5
    result = geron_dpo(pi, pi_ref, beta=beta)
    assert isinstance(result, dict)
    import math
    assert math.isfinite(result["loss"])
    assert result["n"] == B
    # with default preferences all column 0 is chosen, prob_preferred
    # in [0, 1], accuracy a probability
    assert 0.0 <= result["accuracy"] <= 1.0
    assert all(0.0 <= p <= 1.0 for p in result["prob_preferred"])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmdpo as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
