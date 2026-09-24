"""Tests for hmsft.geron_sft."""

import math

from morie.fn import _array_core as np

from morie.fn.hmsft import geron_sft


def test_hmsft_basic():
    """Test basic functionality with instruction-response pairs."""
    instruction_data = [
        ("translate hello", "bonjour"),
        ("translate world", "monde"),
        ("translate car", "voiture"),
        ("translate house", "maison"),
        ("translate cat", "chat"),
        ("translate dog", "chien"),
        ("summarise text", "resume"),
        ("summarise paragraph", "sommaire"),
        ("summarise article", "abrege"),
    ]
    epochs = 200
    lr = 0.5
    result = geron_sft(None, instruction_data, epochs, lr)
    assert isinstance(result, dict)
    expected_keys = {
        "W", "loss", "sum_loss", "loss_curve", "accuracy",
        "predicted", "vocab", "labels", "estimate", "n", "method",
    }
    assert expected_keys.issubset(set(result.keys()))
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["sum_loss"])
    assert 0.0 <= float(result["accuracy"]) <= 1.0
    assert len(result["loss_curve"]) >= 1
    assert int(result["n"]) > 0
    assert len(result["predicted"]) == int(result["n"])


def test_hmsft_edge():
    """Test edge case with minimal but valid input (2 distinct responses)."""
    instruction_data = [
        ("translate hello", "bonjour"),
        ("translate world", "monde"),
        ("summarise text", "resume"),
        ("summarise document", "sommaire"),
    ]
    epochs = 50
    lr = 0.5
    l2 = 0.0
    result = geron_sft(None, instruction_data, epochs, lr, l2)
    assert isinstance(result, dict)
    assert "loss_curve" in result
    assert len(result["loss_curve"]) >= 1
    assert math.isfinite(result["loss"])
    assert float(result["loss"]) >= 0.0
    assert 0.0 <= float(result["accuracy"]) <= 1.0
    assert int(result["n"]) == len(instruction_data)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmsft as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
