"""Tests for ebalw.entropy_balancing."""

from morie.fn import _array_core as np

from morie.fn.ebalw import entropy_balancing


def test_ebalw_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_t = np.random.default_rng(43)
    X = rng_x.normal(0, 1, (100, 5))
    T = rng_t.integers(0, 2, 100)
    result = entropy_balancing(X, T, moments=1)
    assert isinstance(result, dict)
    assert "weights" in result
    assert "balance_achieved" in result
    assert "max_imbalance" in result
    assert "ess" in result
    assert "converged" in result
    w = result["weights"]
    assert w.shape == (int((T == 0).sum()),)
    assert float(w.min()) > 0
    assert abs(float(w.sum()) - 1.0) < 1e-9
    assert result["balance_achieved"] is True
    assert float(result["max_imbalance"]) < 1e-6
    t0 = T == 0
    t1 = T == 1
    target_mean = X[t1].mean(axis=0)
    balanced_mean = (w[:, None] * X[t0]).sum(axis=0)
    max_abs_diff = float(np.max(np.abs(balanced_mean - target_mean)))
    assert max_abs_diff < 1e-6


def test_ebalw_edge():
    """Test edge cases."""
    rng_x = np.random.default_rng(42)
    rng_t = np.random.default_rng(43)
    X = rng_x.normal(0, 1, (100, 5))
    T = rng_t.integers(0, 2, 100)
    result = entropy_balancing(X, T, moments=2)
    assert isinstance(result, dict)
    assert "weights" in result
    w = result["weights"]
    assert float(w.min()) > 0
    assert abs(float(w.sum()) - 1.0) < 1e-9
    t0 = T == 0
    t1 = T == 1
    target_mean = X[t1].mean(axis=0)
    target_var = (X[t1] ** 2).mean(axis=0)
    balanced_mean = (w[:, None] * X[t0]).sum(axis=0)
    balanced_var = (w[:, None] * (X[t0] ** 2)).sum(axis=0)
    assert float(np.max(np.abs(balanced_mean - target_mean))) < 1e-6
    assert float(np.max(np.abs(balanced_var - target_var))) < 1e-6
    assert int(result["n_constraints"]) == 2 * X.shape[1]
