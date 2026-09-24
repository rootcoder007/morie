"""Tests for coxfrl.cox_frailty."""

from morie.fn import _array_core as np

from morie.fn.coxfrl import cox_frailty


def test_coxfrl_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    p = 3
    # 10 clusters of 10 subjects each (integer labels)
    k_labels = np.array([i // 10 for i in range(n)])
    X = rng.normal(0, 1, (n, p))
    # Positive survival times
    time = np.array(rng.uniform(0.1, 10.0, n))
    # Event indicators (0 or 1)
    event = np.array([float(x) for x in rng.integers(0, 2, n)])
    result = cox_frailty(time, event, X, k_labels)
    assert isinstance(result, dict)
    assert "beta" in result
    assert "theta" in result
    assert "n_clusters" in result


def test_coxfrl_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    # 4 clusters of 10 subjects each
    k_labels = np.array([i // 10 for i in range(n)])
    X = rng.normal(0, 1, (n, p))
    # Positive survival times
    time = np.array(rng.uniform(0.1, 10.0, n))
    # Event indicators
    event = np.array([float(x) for x in rng.integers(0, 2, n)])
    result = cox_frailty(time, event, X, k_labels)
    assert isinstance(result, dict)
    assert "frailty" in result
    assert "converged" in result
