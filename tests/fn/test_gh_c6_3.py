"""Tests for gh_c6_3.ghosal_doob_consist."""

from morie.fn import _array_core as np

from morie.fn.gh_c6_3 import ghosal_doob_consist


def test_gh_c6_3_basic():
    """Test basic functionality."""
    theta0 = 0.4
    n = 1500
    seed = 42
    result = ghosal_doob_consist(theta0=theta0, n=n, seed=seed)
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(estimate)

    # Replay the Bernoulli/Beta-posterior mean path with a fresh RNG using the same seed.
    rng = np.random.default_rng(seed)
    S = 0
    path = []
    for i in range(1, n + 1):
        S += 1 if float(rng.uniform(0, 1)) < theta0 else 0
        if i % (n // 10) == 0:
            path.append((1.0 + S) / (2.0 + i))
    expected_estimate = path[-1]
    expected_path = list(path)
    expected_final_error = abs(path[-1] - theta0)

    assert abs(estimate - expected_estimate) < 1e-12
    assert "posterior_mean_path" in result
    returned_path = list(result["posterior_mean_path"])
    assert len(returned_path) == len(expected_path)
    for got, exp in zip(returned_path, expected_path):
        assert abs(float(got) - float(exp)) < 1e-12
    assert "final_error" in result
    assert abs(float(result["final_error"]) - expected_final_error) < 1e-12
    assert result["method"] == "Doob martingale consistency (GvdV 2017 Thm 6.9)"


def test_gh_c6_3_edge():
    """Test edge cases."""
    theta0 = 0.4
    n = 1500
    seed = 7
    result = ghosal_doob_consist(theta0=theta0, n=n, seed=seed)
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(estimate)

    # Independent reproduction of the documented Beta-Bernoulli posterior mean at the last recorded step.
    rng = np.random.default_rng(seed)
    S = 0
    path = []
    for i in range(1, n + 1):
        S += 1 if float(rng.uniform(0, 1)) < theta0 else 0
        if i % (n // 10) == 0:
            path.append((1.0 + S) / (2.0 + i))
    expected_estimate = path[-1]

    assert abs(estimate - expected_estimate) < 1e-12
    # Path length must be exactly 10 (one checkpoint per n//10 interval over [1, n]).
    assert len(list(result["posterior_mean_path"])) == 10
    # Final error should equal |estimate - theta0| by the documented formula.
    assert abs(float(result["final_error"]) - abs(estimate - theta0)) < 1e-12
