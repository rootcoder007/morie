"""Tests for datasb.data_subset_refutation."""

from morie.fn import _array_core as np

from morie.fn.datasb import data_subset_refutation


def _toy_estimator(y, d, X):
    return float(np.mean(y))


def test_datasb_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 2))
    d = (rng.uniform(size=n) < 0.5).astype(float)
    y = 2.0 * d + rng.normal(size=n)
    result = data_subset_refutation(_toy_estimator, y, d, X, n_sims=10, seed=0)
    assert isinstance(result, dict)
    # documented return keys
    for key in ("original", "subset_mean", "subset_sd",
                "relative_change", "passed",
                "excess_variability", "max_single_row_influence"):
        assert key in result

    # independent recomputation of subset_mean from documented behaviour:
    # draw n_sims subsets of size round(fraction*n) (min 3) without
    # replacement and average the estimator over them.
    n_sims = 10
    fraction = 0.8
    k = max(int(round(fraction * n)), 3)
    sub_rng = np.random.default_rng(0)
    vals = []
    for _ in range(n_sims):
        idx = sub_rng.choice(n, size=k, replace=False)
        vals.append(_toy_estimator(y[idx], d[idx], X[idx]))
    expected_mean = float(np.mean(vals))

    assert abs(result["subset_mean"] - expected_mean) < 1e-9


def test_datasb_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 2))
    d = (rng.uniform(size=n) < 0.5).astype(float)
    y = 2.0 * d + rng.normal(size=n)
    result = data_subset_refutation(_toy_estimator, y, d, X, n_sims=5, seed=0)
    assert isinstance(result, dict)
    assert "subset_values" in result
    assert result["n_sims"] == 5
    assert result["n"] == n
