"""Tests for covbal.covariate_balance_check."""

from morie.fn import _array_core as np

from morie.fn.covbal import covariate_balance_check


def test_covbal_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_t = np.random.default_rng(43)
    rng_w = np.random.default_rng(45)
    A = rng_x.normal(0, 1, (100, 10))
    H = (rng_t.random(100) < 0.5).astype(float)
    weights = rng_w.exponential(1, 100)
    result = covariate_balance_check(A, H, weights)
    assert isinstance(result, dict)
    assert "smd_before" in result
    assert "smd_after" in result
    assert "variance_ratio" in result
    assert "n_imbalanced" in result
    assert "balanced" in result
    assert "worst" in result
    assert "threshold" in result
    assert "method" in result

    # Shape checks: one SMD / variance ratio per covariate (column).
    p = A.shape[1]
    assert result["smd_before"].shape == (p,)
    assert result["smd_after"].shape == (p,)
    assert result["variance_ratio"].shape == (p,)

    # Independent arithmetic check of smd_after on a controlled sub-problem.
    # Build a tiny dataset where the weighted means and unweighted pooled sd
    # can be worked out by hand.
    Xs = np.array([[0.0], [2.0], [1.0], [3.0]])
    ts = np.array([1.0, 1.0, 0.0, 0.0])
    ws = np.array([2.0, 2.0, 1.0, 1.0])
    # Treated weighted mean = (0*2 + 2*2) / (2+2) = 1.0
    # Control weighted mean = (1*1 + 3*1) / (1+1) = 2.0
    # Unweighted treated var (ddof=1): mean=1, ((0-1)^2+(2-1)^2)/1 = 2
    # Unweighted control var (ddof=1): mean=2, ((1-2)^2+(3-2)^2)/1 = 2
    # pooled = sqrt((2+2)/2) = sqrt(2)
    pooled = np.sqrt(2.0)
    expected_smd_after = (1.0 - 2.0) / pooled
    r2 = covariate_balance_check(Xs, ts, weights=ws)
    assert abs(float(r2["smd_after"][0]) - expected_smd_after) < 1e-12

    # smd_before with the same dataset using unweighted means.
    expected_smd_before = (1.0 - 2.0) / pooled
    assert abs(float(r2["smd_before"][0]) - expected_smd_before) < 1e-12

    # variance_ratio: both unweighted variances are 2, so ratio is 1.
    assert abs(float(r2["variance_ratio"][0]) - 1.0) < 1e-12

    # n_imbalanced and balanced must be consistent.
    assert int(result["n_imbalanced"]) == int(
        np.sum(np.abs(result["smd_after"]) > result["threshold"])
    )
    assert bool(result["balanced"]) == (int(result["n_imbalanced"]) == 0)

    # worst is the index of the covariate with the largest |smd_after|.
    expected_worst = int(np.argmax(np.abs(result["smd_after"])))
    assert result["worst"] == expected_worst


def test_covbal_edge():
    """Test edge cases."""
    rng_x = np.random.default_rng(42)
    rng_t = np.random.default_rng(43)
    rng_w = np.random.default_rng(45)
    A = rng_x.normal(0, 1, (100, 10))
    H = (rng_t.random(100) < 0.5).astype(float)
    weights = rng_w.exponential(1, 100)
    result = covariate_balance_check(A, H, weights)
    assert isinstance(result, dict)
    assert "smd_before" in result
    assert "smd_after" in result
    assert "variance_ratio" in result
    assert "n_imbalanced" in result
    assert "balanced" in result
    assert "worst" in result
