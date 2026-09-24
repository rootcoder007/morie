"""Tests for kmrhf.kamath_rlhf_pipeline."""

import doctest as _doctest

import pytest

from morie.fn import _array_core as np

from morie.fn.kmrhf import kamath_rlhf_pipeline


def test_kmrhf_basic():
    """Test basic functionality."""
    demos = ["d1", "d2", "d3"]
    preferences = [("a", "b"), ("c", "d"), ("e", "f")]
    pi0 = "pi0"
    result = kamath_rlhf_pipeline(
        demos, preferences, pi0,
        sft=lambda p, d: p + "+sft",
        train_rm=lambda prefs: (lambda y: len(y)),
        ppo=lambda pi, rm, ref: pi + "+ppo")
    assert isinstance(result, dict)
    assert result["policy"] == "pi0+sft+ppo"
    assert result["policy_sft"] == "pi0+sft"
    assert result["kl_reference_is_sft"] is True
    assert result["stages"] == ["sft", "reward_model", "ppo"]
    assert result["n_demos"] == 3
    assert result["n_preferences"] == 3
    assert result["method"] == "RLHF pipeline SFT -> RM -> PPO (KL anchored to SFT)"


def test_kmrhf_edge():
    """Test edge cases: empty inputs raise ValueError."""
    sft = lambda p, d: p + "+sft"
    train_rm = lambda prefs: (lambda y: len(y))
    ppo = lambda pi, rm, ref: pi + "+ppo"
    with pytest.raises(ValueError):
        kamath_rlhf_pipeline(
            [], [("a", "b")], "pi0",
            sft=sft, train_rm=train_rm, ppo=ppo)
    with pytest.raises(ValueError):
        kamath_rlhf_pipeline(
            ["d1"], [], "pi0",
            sft=sft, train_rm=train_rm, ppo=ppo)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import morie.fn.kmrhf as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
