"""Tests for hrzlew.horowitz_lewbel_estimator."""

from morie.fn import _array_core as np

from morie.fn.hrzlew import horowitz_lewbel_estimator


def test_hrzlew_basic():
    """Test basic functionality with nonparametric density."""
    rng_x = np.random.default_rng(42)
    rng_z = np.random.default_rng(44)
    n = 40
    d = 3
    x = rng_x.normal(0, 1, (n, d))
    z = rng_z.normal(0, 1, n)
    # Generate binary y from a simple latent variable to ensure variation
    latent = [z[i] + x[i][0] for i in range(n)]
    y = [1 if latent[i] > 0 else 0 for i in range(n)]
    bandwidth = 0.3
    result = horowitz_lewbel_estimator(x, y, z, bandwidth)
    assert isinstance(result, dict)
    # Verify that all expected keys are present
    expected_keys = [
        "beta", "se", "coefficient_on_V", "min_density", "max_weight",
        "root_n_consistent", "heteroskedasticity_allowed",
        "identifies_choice_probabilities", "bandwidth", "endogenous",
        "n", "d", "method"
    ]
    for key in expected_keys:
        assert key in result
    # coefficient_on_V is fixed at 1.0
    assert result["coefficient_on_V"] == 1.0
    # These booleans are fixed
    assert result["root_n_consistent"] is True
    assert result["heteroskedasticity_allowed"] is True
    assert result["identifies_choice_probabilities"] is True
    # Without instruments, endogenous is False
    assert result["endogenous"] is False
    # n and d
    assert result["n"] == n
    assert result["d"] in (d, d + 1)
    # beta and se have length matching the design matrix after constant addition
    assert len(result["beta"]) in (d, d + 1)
    assert len(result["se"]) == len(result["beta"])
    # bandwidth should be the value we provided
    assert result["bandwidth"] == bandwidth
    # min_density and max_weight are numeric
    assert isinstance(result["min_density"], (int, float))
    assert isinstance(result["max_weight"], (int, float))


def test_hrzlew_edge():
    """Test edge case with small sample and default bandwidth."""
    rng_x = np.random.default_rng(42)
    rng_z = np.random.default_rng(44)
    n = 20
    d = 2
    x = rng_x.normal(0, 1, (n, d))
    z = rng_z.normal(0, 1, n)
    # Generate binary y from a simple latent variable to ensure variation
    latent = [z[i] + x[i][0] for i in range(n)]
    y = [1 if latent[i] > 0 else 0 for i in range(n)]
    # Call without specifying bandwidth (uses default)
    result = horowitz_lewbel_estimator(x, y, z)
    assert isinstance(result, dict)
    # Check keys
    expected_keys = [
        "beta", "se", "coefficient_on_V", "min_density", "max_weight",
        "root_n_consistent", "heteroskedasticity_allowed",
        "identifies_choice_probabilities", "bandwidth", "endogenous",
        "n", "d", "method"
    ]
    for key in expected_keys:
        assert key in result
    # coefficient_on_V fixed
    assert result["coefficient_on_V"] == 1.0
    # n and d
    assert result["n"] == n
    assert result["d"] in (d, d + 1)
    # beta length
    assert len(result["beta"]) in (d, d + 1)
    # endogenous is False when no instruments
    assert result["endogenous"] is False
