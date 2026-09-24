"""Tests for grkmo.geron_kmeans_objective."""

from morie.fn import _array_core as np

from morie.fn.grkmo import geron_kmeans_objective


def test_grkmo_basic():
    """Test basic functionality."""
    X = [[0.0, 0.0], [3.0, 4.0]]
    centroids = [[0.0, 0.0]]
    labels = [0, 0]
    result = geron_kmeans_objective(X, centroids, labels)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grkmo_edge():
    """Test edge cases."""
    X = [[0.0, 0.0], [3.0, 4.0]]
    centroids = [[0.0, 0.0]]
    labels = [0, 0]
    result = geron_kmeans_objective(X, centroids, labels)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grkmo as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
