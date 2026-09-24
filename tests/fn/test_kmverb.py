"""Tests for kmverb.kamath_verbalizer_mapping."""

from morie.fn import _array_core as np

import pytest

from morie.fn.kmverb import kamath_verbalizer_mapping


def test_kmverb_basic():
    """Test basic functionality using the docstring example."""
    logits = [0.0, 0.0, 0.0, 0.0]
    vocab = ["a", "b", "c", "d"]
    verbalizer_map = {"pos": ["a", "b"], "neg": ["c"]}
    result = kamath_verbalizer_mapping(logits, vocab, verbalizer_map)
    # result is a dict-like object
    assert isinstance(result, dict)
    # check keys
    expected_keys = {"probabilities", "normalized", "prediction", "mass_on_labels", "mass_outside"}
    assert expected_keys.issubset(result.keys())
    # probabilities: pos = 0.5, neg = 0.25
    assert abs(result["probabilities"]["pos"] - 0.5) < 1e-12
    assert abs(result["probabilities"]["neg"] - 0.25) < 1e-12
    # prediction is the class with highest probability
    assert result["prediction"] == "pos"
    # normalized probabilities sum to 1
    norm_sum = sum(result["normalized"].values())
    assert abs(norm_sum - 1.0) < 1e-12
    # mass_outside is 1 - mass_on_labels
    assert abs(result["mass_on_labels"] + result["mass_outside"] - 1.0) < 1e-12
    # mass_outside equals 0.25
    assert abs(result["mass_outside"] - 0.25) < 1e-12
    # normalized['pos'] is 2/3
    assert abs(result["normalized"]["pos"] - 2 / 3) < 1e-12


def test_kmverb_edge():
    """Test that overlapping label sets raise ValueError."""
    vocab = ["a", "b", "c"]
    logits = [0.0, 0.0, 0.0]
    verbalizer_map = {"pos": ["a", "b"], "neg": ["b"]}
    with pytest.raises(ValueError):
        kamath_verbalizer_mapping(logits, vocab, verbalizer_map)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmverb as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
