"""Tests for kmfst.kamath_fasttext_subword."""

from morie.fn import _array_core as np

from morie.fn.kmfst import kamath_fasttext_subword


def test_kmfst_basic():
    """Test basic functionality."""
    word = "ab"
    tbl = {"<a": [1.0, 0.0], "ab": [0.0, 1.0], "b>": [1.0, 1.0], "<ab>": [0.0, 0.0]}
    n_min = 2
    n_max = 2
    result = kamath_fasttext_subword(word, tbl, n_min, n_max)
    assert isinstance(result, dict)
    assert "vector" in result
    assert "estimate" in result
    assert "n_known" in result
    assert "n_missing" in result
    # From the docstring worked example
    assert result["vector"] == [2.0, 2.0]
    assert result["n_known"] == 4
    assert result["n_missing"] == 0


def test_kmfst_edge():
    """Test edge cases."""
    word = "ab"
    # Partial table -- some subwords are missing from the lookup
    tbl = {"<a": [1.0, 0.0], "ab": [0.0, 1.0]}
    n_min = 2
    n_max = 2
    result = kamath_fasttext_subword(word, tbl, n_min, n_max)
    assert isinstance(result, dict)
    assert "vector" in result
    assert "n_known" in result
    assert "n_missing" in result
    # At least one known and at least one missing
    assert result["n_known"] >= 1
    assert result["n_missing"] >= 1
    # Vector has the embedding dim of the known entries
    assert len(result["vector"]) == 2


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmfst as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
