"""Tests for kmpet.kamath_pet_loss."""

from morie.fn import _array_core as np

from morie.fn.kmpet import kamath_pet_loss


def test_kmpet_basic():
    """Test basic functionality."""
    verbalizer_logits = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y_true = 0.5
    mlm_logits = 0.5
    mlm_targets = 0.5
    alpha = 0.5
    result = kamath_pet_loss(verbalizer_logits, y_true, mlm_logits, mlm_targets, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_kmpet_edge():
    """Test edge cases."""
    verbalizer_logits = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y_true = 0.5
    mlm_logits = 0.5
    mlm_targets = 0.5
    alpha = 0.5
    result = kamath_pet_loss(verbalizer_logits, y_true, mlm_logits, mlm_targets, alpha)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmpet as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
