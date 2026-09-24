"""Tests for hmql.geron_q_learning."""

import math

from morie.fn import _array_core as np

from morie.fn.hmql import geron_q_learning


def test_hmql_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_states, n_actions = 5, 3
    Q = rng.normal(0, 1, (n_states, n_actions))
    s = 0
    a = 1
    r = 1.5
    s_next = 2
    alpha = 0.5
    gamma = 0.9
    result = geron_q_learning(Q, s, a, r, s_next, alpha, gamma)
    assert isinstance(result, dict)
    for key in ("Q", "td_error", "target", "old_value", "new_value",
                "estimate", "n", "method"):
        assert key in result
    # The returned Q table keeps the input shape.
    assert len(result["Q"]) == n_states
    assert all(len(row) == n_actions for row in result["Q"])
    # Non-terminal target = r + gamma * max_a' Q(s_next, a').
    expected_target = r + gamma * max(Q[s_next])
    assert math.isclose(float(result["target"]), expected_target, rel_tol=1e-9)
    # Old value is the entry that was overwritten.
    assert math.isclose(float(result["old_value"]), float(Q[s][a]), rel_tol=1e-9)
    # td_error and the Bellman update must agree with target / old_value.
    assert math.isclose(
        float(result["td_error"]),
        float(result["target"]) - float(result["old_value"]),
        rel_tol=1e-9,
    )
    assert math.isclose(
        float(result["new_value"]),
        float(result["old_value"]) + alpha * float(result["td_error"]),
        rel_tol=1e-9,
    )
    assert math.isfinite(float(result["target"]))
    assert math.isfinite(float(result["td_error"]))
    assert math.isfinite(float(result["new_value"]))


def test_hmql_edge():
    """Test terminal transition drops the bootstrap target."""
    rng = np.random.default_rng(42)
    n_states, n_actions = 4, 2
    Q = rng.normal(0, 1, (n_states, n_actions))
    s = 0
    a = 0
    r = 2.0
    s_next = 3
    alpha = 1.0
    gamma = 0.9
    result = geron_q_learning(Q, s, a, r, s_next, alpha, gamma, done=True)
    assert isinstance(result, dict)
    # On a terminal step the bootstrap is dropped, so target == r
    # regardless of Q(s_next, .) or gamma.
    assert math.isclose(float(result["target"]), r, rel_tol=1e-9)
    # With alpha == 1 on a terminal step, new_value collapses to r.
    assert math.isclose(float(result["new_value"]), r, rel_tol=1e-9)
    assert math.isfinite(float(result["new_value"]))
    # Updated Q table is still well-shaped.
    assert len(result["Q"]) == n_states
    assert all(len(row) == n_actions for row in result["Q"])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmql as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
