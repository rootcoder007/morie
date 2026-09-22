"""Tests for dprrep.randomized_response_dp."""

from morie.fn import _array_core as np

from morie.fn.dprrep import randomized_response_dp


def test_dprrep_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    truth = (rng.random(100) < 0.3).astype(int)
    epsilon = 1.0
    result = randomized_response_dp(truth, epsilon=epsilon)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result

    # Validate the documented fields are all present.
    for key in ("responses", "p_truth", "raw_proportion", "estimate", "se"):
        assert key in result

    # p_truth must equal the logistic of epsilon (documented formula).
    expected_p = float(np.exp(epsilon) / (1.0 + np.exp(epsilon)))
    assert abs(float(result["p_truth"]) - expected_p) < 1e-12

    # Debiased estimate formula: (raw_proportion - (1-p)) / (2p - 1).
    raw = float(result["raw_proportion"])
    est = float(result["estimate"])
    expected_est = (raw - (1.0 - expected_p)) / (2.0 * expected_p - 1.0)
    assert abs(est - expected_est) < 1e-12

    # Responses must be binary.
    resp = np.asarray(result["responses"]).ravel()
    assert resp.size == truth.size
    assert bool(np.all((resp == 0) | (resp == 1)))


def test_dprrep_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    truth = (rng.random(100) < 0.3).astype(int)
    epsilon = 1.0
    result = randomized_response_dp(truth, epsilon=epsilon)
    assert isinstance(result, dict)

    # The debiased estimate must be finite and not trivially the raw mean.
    est = float(result["estimate"])
    raw = float(result["raw_proportion"])
    assert est != raw

    # With a deterministic seed the result must be reproducible.
    rng2 = np.random.default_rng(7)
    truth2 = (rng2.random(50) < 0.5).astype(int)
    r1 = randomized_response_dp(truth2, epsilon=1.5, seed=123)
    r2 = randomized_response_dp(truth2, epsilon=1.5, seed=123)
    assert float(r1["raw_proportion"]) == float(r2["raw_proportion"])
