"""Tests for agmurt.muzero_reanalyze_target."""

from morie.fn import _array_core as np

from morie.fn.agmurt import muzero_reanalyze_target


def test_agmurt_basic():
    """Test basic functionality."""
    T = 100
    A = 4
    rng = np.random.default_rng(42)
    rewards = rng.normal(0, 1, T)
    freshvalues = rng.normal(0, 1, T)
    visits = rng.integers(0, 100, (T, A)).astype(float)
    n = 5
    gamma = 0.997
    result = muzero_reanalyze_target(rewards, freshvalues, visits, n=n, gamma=gamma)
    assert isinstance(result, dict)
    assert "target" in result
    assert "policy" in result
    assert "priority" in result
    assert "prob" in result
    assert "weight" in result
    assert result["T"] == T
    assert result["A"] == A
    assert result["n"] == n
    assert result["gamma"] == gamma

    # Independently compute the n-step bootstrapped target from the formula
    expected_target = []
    for t in range(T):
        s = 0.0
        for j in range(n):
            if t + j < T:
                s += (gamma ** j) * rewards[t + j]
        if t + n < T:
            s += (gamma ** n) * freshvalues[t + n]
        expected_target.append(s)
    for actual, expected in zip(result["target"], expected_target):
        assert abs(actual - expected) < 1e-9

    # Independently compute the normalised visit policy
    for t in range(T):
        tot = sum(visits[t])
        for a_idx in range(A):
            expected_p = visits[t][a_idx] / tot
            assert abs(result["policy"][t][a_idx] - expected_p) < 1e-9


def test_agmurt_edge():
    """Test edge cases."""
    T = 50
    A = 3
    rng = np.random.default_rng(7)
    rewards = rng.normal(0, 1, T)
    freshvalues = rng.normal(0, 1, T)
    visits = rng.integers(0, 50, (T, A)).astype(float)
    result = muzero_reanalyze_target(rewards, freshvalues, visits)
    assert isinstance(result, dict)
    assert "target" in result
    assert "policy" in result
    assert "priority" in result
    assert "prob" in result
    assert "weight" in result
    assert result["T"] == T
    assert result["A"] == A
