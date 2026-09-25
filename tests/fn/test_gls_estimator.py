"""Tests for gls_estimator.gls_estimator."""

import pytest

from morie.fn.gls_estimator import (
    gls_estimator,
)


def test_gls_estimator_basic():
    """(X' C^-1 X)^-1 X' C^-1 z recomputed with a diagonal C (weighted
    least squares) and a trend design."""
    X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    v = [1.0, 2.0, 0.5, 4.0]
    z = [1.1, 2.9, 5.2, 6.8]
    C = [[v[i] if i == j else 0.0 for j in range(4)] for i in range(4)]
    r = gls_estimator(X, C, z)
    w = [1 / t for t in v]
    sw, sx, sxx = sum(w), sum(a * b[1] for a, b in zip(w, X)), sum(a * b[1] ** 2 for a, b in zip(w, X))
    sz, sxz = sum(a * c for a, c in zip(w, z)), sum(a * b[1] * c for a, b, c in zip(w, X, z))
    det = sw * sxx - sx * sx
    assert r["values"] == pytest.approx([(sxx * sz - sx * sxz) / det, (sw * sxz - sx * sz) / det], rel=1e-12)


def test_gls_estimator_edge():
    """Two estimates of one mean with variances 1 and 4 combine to 2.6."""
    r = gls_estimator([1.0, 1.0], [[1.0, 0.0], [0.0, 4.0]], [2.0, 5.0])
    assert r["value"] == pytest.approx(2.6, rel=1e-14)
    with pytest.raises(ValueError, match="shape"):
        gls_estimator([1.0, 1.0], [[1.0]], [2.0, 5.0])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import importlib as _importlib

# the package also exports a function of this name, so the import
# statement would bind that function, not the module
_doctest_module = _importlib.import_module("morie.fn.gls_estimator")


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
