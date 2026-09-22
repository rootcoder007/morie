"""Tests for abcsmc.abc_smc_epi."""

from morie.fn import _array_core as np

from morie.fn.abcsmc import abc_smc_epi


def _toy_model(theta):
    # Simple deterministic model: returns a vector whose first component
    # is the value of theta[0] (so the "observed" statistic at 0.7 is
    # recoverable when the prior covers 0.7).
    return [theta[0], 0.0]


def test_abcsmc_basic():
    """Test basic functionality with a simple recoverable model."""
    rng = np.random.default_rng(42)
    # Observed summary statistics: we want the model to match when theta[0] == 0.7.
    summary_stats = [0.7, 0.0]
    priors = [[0.0, 1.0]]
    n_particles = 32
    result = abc_smc_epi(
        _toy_model, summary_stats, priors=priors, n_particles=n_particles,
        schedule=[1.0, 0.5, 0.1], kernel_sd=0.1,
    )

    # Function returns a RichResult with a .payload dict, not a bare dict.
    assert hasattr(result, "payload")
    payload = result.payload

    # Documented return keys.
    assert "estimate" in payload
    assert "theta" in payload
    assert "weights" in payload
    assert "ess" in payload
    assert "accept" in payload

    # Shapes / sizes match the documented contract.
    theta = payload["theta"]
    weights = payload["weights"]
    assert len(theta) == n_particles
    assert len(weights) == n_particles

    # Weights are normalised to sum to 1 (the Kish ESS formula uses this).
    assert abs(sum(weights) - 1.0) < 1e-9

    # Kish ESS: (sum w)^2 / sum(w^2).  Independently computed from weights.
    s1 = sum(weights)
    s2 = sum(w * w for w in weights)
    expected_ess = (s1 * s1) / s2 if s2 > 0.0 else 0.0
    assert abs(payload["ess"] - expected_ess) < 1e-9

    # The weighted mean of parameter 0 (estimate) computed independently
    # from the returned theta and weights must equal payload["estimate"].
    m0 = sum(weights[i] * theta[i][0] for i in range(len(theta)))
    assert abs(payload["estimate"] - m0) < 1e-9

    # All particles lie within the prior support.
    lo, hi = priors[0]
    for th in theta:
        assert lo - 1e-12 <= th[0] <= hi + 1e-12


def test_abcsmc_edge():
    """Test edge cases: minimum n_particles and tight prior around truth."""
    rng = np.random.default_rng(42)
    summary_stats = [0.5, 0.0]
    priors = [[0.0, 1.0]]
    # Smallest reasonable particle count.
    n_particles = 4
    result = abc_smc_epi(
        _toy_model, summary_stats, priors=priors, n_particles=n_particles,
        schedule=[0.5, 0.1], kernel_sd=0.05,
    )
    assert hasattr(result, "payload")
    payload = result.payload
    assert len(payload["theta"]) == n_particles
    assert len(payload["weights"]) == n_particles
    assert abs(sum(payload["weights"]) - 1.0) < 1e-9

    # accept is a list with one entry per attempted population.
    assert isinstance(payload["accept"], list)
    assert len(payload["accept"]) >= 1
    for a in payload["accept"]:
        assert 0.0 <= a <= 1.0
