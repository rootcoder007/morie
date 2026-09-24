"""Tests for hmvf.geron_value_function."""

import math

from morie.fn import _array_core as np
from morie.fn.hmvf import geron_value_function


def test_hmvf_basic():
    """Test basic functionality."""
    n_s, n_a = 3, 2
    rng = np.random.default_rng(42)
    # Build valid transition tensor P[s, a, s'] with rows summing to 1
    P = []
    for s in range(n_s):
        P_a = []
        for a in range(n_a):
            row = rng.uniform(0, 1, n_s)
            total = float(sum(row))
            row = [x / total for x in row]
            P_a.append(row)
        P.append(P_a)
    # Build rewards R[s, a]
    R = rng.normal(0, 1, (n_s, n_a))
    # Build stochastic policy pi[s, a] with rows summing to 1
    pi = []
    for s in range(n_s):
        row = rng.uniform(0, 1, n_a)
        total = float(sum(row))
        row = [x / total for x in row]
        pi.append(row)
    gamma = 0.9
    s = 0
    result = geron_value_function(s, pi, gamma, P=P, R=R)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "V" in result
    assert "residual" in result
    V = result["V"]
    assert len(V) == n_s
    assert math.isfinite(float(result["estimate"]))


def test_hmvf_edge():
    """Test edge cases: deterministic policy with s=None."""
    # Simple deterministic MDP from the docstring example:
    # state 0 always moves to state 1 with reward 1,
    # state 1 absorbs with reward 0. With gamma = 0.9.
    P = [[[0.0, 1.0]], [[0.0, 1.0]]]
    R = [[[0.0, 1.0]], [[0.0, 0.0]]]
    # Deterministic policy as 1-D vector of action indices
    pi = [0, 0]
    gamma = 0.9
    s = None  # means state 0
    result = geron_value_function(s, pi, gamma, P=P, R=R)
    assert isinstance(result, dict)
    assert "V" in result
    assert "estimate" in result
    V = result["V"]
    assert len(V) == 2
    assert math.isfinite(float(result["estimate"]))


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmvf as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
