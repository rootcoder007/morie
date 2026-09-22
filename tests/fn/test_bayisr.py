"""Tests for bayisr.importance_resample."""

from morie.fn import _array_core as np

from morie.fn.bayisr import importance_resample


def test_bayisr_basic():
    """Test basic functionality."""
    rng_prop = np.random.default_rng(42)
    proposal_samples = list(rng_prop.normal(0, 1, 100))

    def log_target(x):
        # log of N(0, 1) up to a constant (matches the proposal here)
        v = float(x)
        return -0.5 * v * v

    def log_proposal(x):
        v = float(x)
        return -0.5 * v * v

    m = 25
    seed = 7
    res = importance_resample(
        proposal_samples, log_target, log_proposal, m, seed
    )

    # The function returns a RichResult (mapping-like), not necessarily a dict.
    assert hasattr(res, "__getitem__")

    # Documented keys
    assert "resample" in res
    assert "indices" in res
    assert "weights" in res
    assert "ess" in res
    assert "n" in res
    assert "m" in res
    assert "seed" in res
    assert "method" in res

    # Shapes / sizes
    assert res["n"] == 100
    assert res["m"] == m
    assert res["seed"] == seed
    assert len(res["resample"]) == m
    assert len(res["indices"]) == m
    assert len(res["weights"]) == 100

    # Indices are valid 0-based into the input draws
    for j in res["indices"]:
        assert 0 <= int(j) < 100

    # Weights are non-negative and sum to 1
    wsum = sum(float(w) for w in res["weights"])
    assert abs(wsum - 1.0) < 1e-12
    assert all(float(w) >= 0.0 for w in res["weights"])

    # ess = 1 / sum(wbar_i^2) where wbar are the normalized weights.
    # With log_target == log_proposal, log-ratios are zero so each unnormalized
    # weight equals 1, hence each normalized weight equals 1/n exactly.
    n = 100
    expected_ess = 1.0 / sum((1.0 / n) ** 2 for _ in range(n))
    assert abs(float(res["ess"]) - expected_ess) < 1e-9

    # Each normalized weight equals 1/n exactly (log_target == log_proposal).
    for w in res["weights"]:
        assert abs(float(w) - (1.0 / n)) < 1e-12

    # Resampled draws are drawn from the original proposal draws.
    proposal_set = {float(x) for x in proposal_samples}
    for x in res["resample"]:
        assert float(x) in proposal_set


def test_bayisr_edge():
    """Test edge cases."""
    rng_prop = np.random.default_rng(42)
    proposal_samples = list(rng_prop.normal(0, 1, 100))

    def log_target(x):
        v = float(x)
        return -0.5 * v * v

    def log_proposal(x):
        v = float(x)
        return -0.5 * v * v

    res = importance_resample(
        proposal_samples, log_target, log_proposal, 10, seed=0
    )
    assert hasattr(res, "__getitem__")
    assert "resample" in res
    assert len(res["resample"]) == 10
