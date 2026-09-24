"""Tests for hmstr.geron_stratified_sampling."""

import math

from morie.fn import _array_core as np

from morie.fn.hmstr import geron_stratified_sampling


def test_hmstr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 100, 5
    X = rng.normal(0, 1, (n, p))
    stratum = list(rng.integers(0, 3, n))
    n_total = 30
    result = geron_stratified_sampling(X, stratum=stratum, n_total=n_total)
    assert isinstance(result, dict)
    for key in ("indices", "X_sample", "allocation", "max_share_error", "n"):
        assert key in result
    assert result["n"] == n_total
    assert sum(result["allocation"].values()) == n_total
    assert math.isfinite(result["max_share_error"])
    assert result["max_share_error"] >= 0.0


def test_hmstr_edge():
    """Test edge cases with small strata from the docstring example."""
    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [0, 0, 0, 0, 1, 1]
    result = geron_stratified_sampling(X, y=y, n_total=3)
    assert isinstance(result, dict)
    assert result["n"] == 3
    assert len(result["indices"]) == 3
    alloc = result["allocation"]
    assert sum(alloc.values()) == 3
    assert max(alloc.values()) == 2
    assert min(alloc.values()) == 1
    assert math.isfinite(result["max_share_error"])
    assert result["max_share_error"] >= 0.0


# --- appended: the module's own worked example as a gate -----------
import doctest as _doctest

import morie.fn.hmstr as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
