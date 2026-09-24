"""Tests for kmmoe.kamath_moe_router_softmax."""

import math

from morie.fn import _array_core as np

from morie.fn.kmmoe import kamath_moe_router_softmax


def _make_experts(n):
    """Return n callable experts; the i-th one returns (i+1)*v[0]."""
    return [lambda v, i=i: v[0] * (i + 1) for i in range(n)]


def test_kmmoe_basic():
    """Basic MoE call: random router, small d, k=2 of 5 experts."""
    rng = np.random.default_rng(42)
    n_experts = 5
    dim_x = 3
    k = 2

    x = list(rng.normal(0.0, 1.0, dim_x))
    Wr = [list(rng.normal(0.0, 1.0, n_experts)) for _ in range(dim_x)]
    experts = _make_experts(n_experts)

    result = kamath_moe_router_softmax(x, Wr, experts, k)

    expected_keys = ("output", "gate_weights", "selected_experts",
                     "n_active", "experts_evaluated", "estimate",
                     "k", "n", "method")
    for key in expected_keys:
        assert key in result

    assert result["n"] == n_experts
    assert result["k"] == k
    assert result["n_active"] == k
    assert len(result["gate_weights"]) == n_experts
    assert len(result["selected_experts"]) == k
    assert result["experts_evaluated"] == k
    assert math.isfinite(result["estimate"])
    assert all(w >= 0.0 for w in result["gate_weights"])
    # Top-k renormalised softmax: nonzero weights sum to 1.
    assert abs(sum(result["gate_weights"]) - 1.0) < 1e-9


def test_kmmoe_edge():
    """Edge case: k equals n_experts so every expert is active."""
    rng = np.random.default_rng(7)
    n_experts = 3
    dim_x = 2
    k = n_experts

    x = list(rng.normal(0.0, 1.0, dim_x))
    Wr = [list(rng.normal(0.0, 1.0, n_experts)) for _ in range(dim_x)]
    experts = _make_experts(n_experts)

    result = kamath_moe_router_softmax(x, Wr, experts, k)

    expected_keys = ("output", "gate_weights", "selected_experts",
                     "n_active", "experts_evaluated", "estimate",
                     "k", "n", "method")
    for key in expected_keys:
        assert key in result

    assert result["n"] == n_experts
    assert result["k"] == k
    assert result["n_active"] == n_experts
    assert result["experts_evaluated"] == n_experts
    assert math.isfinite(result["estimate"])
    # k == n means the gate is the full softmax; weights sum to 1.
    assert abs(sum(result["gate_weights"]) - 1.0) < 1e-9


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmmoe as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
