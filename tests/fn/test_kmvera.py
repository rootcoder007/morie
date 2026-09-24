"""Tests for kmvera.kamath_vera_adapter."""

from morie.fn import _array_core as np

from morie.fn.kmvera import kamath_vera_adapter


def test_kmvera_basic():
    """Test basic functionality."""
    d, k, r = 5, 4, 3
    W0 = np.random.default_rng(42).normal(0, 1, (d, k))
    A_frozen = np.random.default_rng(43).normal(0, 1, (r, k))
    B_frozen = np.random.default_rng(44).normal(0, 1, (d, r))
    lam_b = np.random.default_rng(45).normal(0, 1, d)
    lam_d = np.random.default_rng(46).normal(0, 1, r)
    x = np.random.default_rng(47).normal(0, 1, k)
    result = kamath_vera_adapter(W0, A_frozen, B_frozen, lam_b, lam_d, x)
    assert "h" in result
    assert "base" in result
    assert "delta" in result
    assert "estimate" in result
    assert "rank" in result
    assert "n_trainable" in result
    assert "n_trainable_lora_equivalent" in result
    assert "n_frozen" in result
    assert "n" in result
    assert "method" in result
    assert len(result["h"]) == d
    assert len(result["base"]) == d
    assert len(result["delta"]) == d
    assert result["rank"] == r
    assert result["n_trainable"] == d + r
    assert result["n_trainable_lora_equivalent"] == r * k + d * r
    assert result["n_frozen"] == d * k + r * k + d * r
    assert result["n"] == d
    assert result["method"] == "VeRA h = W0 x + Lambda_b B Lambda_d A x"


def test_kmvera_edge():
    """Test edge cases."""
    d, k, r = 2, 3, 1
    W0 = np.random.default_rng(10).normal(0, 1, (d, k))
    A_frozen = np.random.default_rng(11).normal(0, 1, (r, k))
    B_frozen = np.random.default_rng(12).normal(0, 1, (d, r))
    lam_b = np.random.default_rng(13).normal(0, 1, d)
    lam_d = np.random.default_rng(14).normal(0, 1, r)
    x = np.random.default_rng(15).normal(0, 1, k)
    result = kamath_vera_adapter(W0, A_frozen, B_frozen, lam_b, lam_d, x)
    assert "h" in result
    assert len(result["h"]) == d
    assert result["rank"] == r
    assert result["n_trainable"] == d + r


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmvera as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
