"""Tests for kmhyde.kamath_hyde_hypothetical_doc."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.kmhyde import kamath_hyde_hypothetical_doc


def test_kmhyde_basic():
    """Test basic functionality with a callable model returning a vector."""
    rng = np.random.default_rng(42)
    # Create a small corpus: 5 documents, each of dimension 3.
    D = [list(rng.normal(0, 1, 3)) for _ in range(5)]
    # Model returns a hypothetical vector of correct width.
    model = lambda q: [1.0, 0.0, 0.0]
    result = kamath_hyde_hypothetical_doc("what colour", model, D, k=3)
    assert isinstance(result, dict)
    # Check that the returned RichResult contains the expected keys.
    assert "retrieved" in result
    assert isinstance(result["retrieved"], list)
    assert len(result["retrieved"]) == 3
    assert "similarities" in result
    assert isinstance(result["similarities"], list)
    assert len(result["similarities"]) == 3
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["k"] == 3
    assert result["n"] == 5
    assert result["hypothetical"] == [1.0, 0.0, 0.0]


def test_kmhyde_edge():
    """Test that k larger than the number of documents raises ValueError."""
    rng = np.random.default_rng(42)
    D = [list(rng.normal(0, 1, 3)) for _ in range(2)]
    model = lambda q: [1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        kamath_hyde_hypothetical_doc("q", model, D, k=3)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmhyde as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
