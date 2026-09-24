"""Tests for kmtopk.kamath_moe_top_k_gating."""

from morie.fn import _array_core as np

from morie.fn.kmtopk import kamath_moe_top_k_gating


def test_kmtopk_basic():
    """Test basic functionality with non-negative gate scores."""
    rng = np.random.default_rng(42)
    n = 100
    gates = rng.uniform(0.0, 1.0, n)
    k = 5
    result = kamath_moe_top_k_gating(gates, k)
    assert isinstance(result, dict)
    for key in ("weights", "selected_experts", "kept_mass", "n_active",
                "estimate", "k", "n", "method"):
        assert key in result
    assert len(result["weights"]) == n
    assert len(result["selected_experts"]) == k
    assert result["k"] == k
    assert result["n"] == n
    assert result["n_active"] == k
    assert abs(sum(result["weights"]) - 1.0) < 1e-12
    assert all(v >= 0 for v in result["weights"])
    assert sum(1 for v in result["weights"] if v > 0) == k


def test_kmtopk_edge():
    """Test edge case k=1 (only the top expert is selected)."""
    rng = np.random.default_rng(42)
    n = 20
    gates = rng.uniform(0.0, 1.0, n)
    k = 1
    result = kamath_moe_top_k_gating(gates, k)
    assert isinstance(result, dict)
    assert len(result["weights"]) == n
    assert len(result["selected_experts"]) == 1
    assert result["k"] == 1
    assert result["n"] == n
    assert result["n_active"] == 1
    assert abs(sum(result["weights"]) - 1.0) < 1e-12
    nonzero = [v for v in result["weights"] if v > 0]
    assert len(nonzero) == 1
    assert abs(nonzero[0] - 1.0) < 1e-12


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmtopk as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
