"""Tests for hmnmf.geron_nmf."""

import math

from morie.fn import _array_core as np

from morie.fn.hmnmf import geron_nmf


def test_hmnmf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 1.0, (40, 3))
    n_components = 2
    result = geron_nmf(X, n_components)
    assert isinstance(result, dict)
    for key in ("W", "H", "reconstruction", "reconstruction_error",
                "relative_error", "n_iter", "estimate", "n", "method"):
        assert key in result
    W = result["W"]
    H = result["H"]
    recon = result["reconstruction"]
    rel = result["relative_error"]
    assert W.shape == (40, 2)
    assert H.shape == (2, 3)
    assert recon.shape == (40, 3)
    assert np.all(W >= 0)
    assert np.all(H >= 0)
    assert math.isfinite(rel)


def test_hmnmf_edge():
    """Test edge cases."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (10, 4))
    result = geron_nmf(X, n_components=1)
    assert isinstance(result, dict)
    assert result["W"].shape == (10, 1)
    assert result["H"].shape == (1, 4)
    assert result["reconstruction"].shape == (10, 4)
    assert np.all(result["W"] >= 0)
    assert np.all(result["H"] >= 0)
    assert math.isfinite(result["relative_error"])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmnmf as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
