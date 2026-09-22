"""Tests for abcrej.abc_rejection."""

from morie.fn import _array_core as np

from morie.fn.abcrej import abc_rejection


def _make_sim(slope, intercept):
    """A deterministic simulator: returns intercept + slope * theta[0]."""
    def sim(theta, rng):
        return np.asarray([intercept + slope * theta[0]])
    return sim


def test_abcrej_basic():
    """Test basic functionality with a 1-D simulator and 1-D summaries."""
    rng = np.random.default_rng(0)
    obs = np.asarray([0.5], dtype=float)

    slope = 2.0
    intercept = -1.0
    sim = _make_sim(slope, intercept)

    prior = [(0.0, 1.0)]
    eps = 0.05
    n_draws = 500
    seed = 123

    result = abc_rejection(sim, obs, eps, prior, n_draws=n_draws, seed=seed)

    # Result must be a dict-like RichResult with the documented keys.
    assert isinstance(result, dict)

    # The function always returns these keys (see docstring).
    assert "samples" in result
    assert "n_accepted" in result
    assert "acceptance_rate" in result
    assert "distances" in result
    assert "posterior_mean" in result

    k = result["n_accepted"]
    assert isinstance(k, int)
    assert 0 <= k <= n_draws
    assert k == len(result["samples"])
    assert k == len(result["distances"])

    # Independent check: acceptance_rate == n_accepted / n_draws.
    expected_rate = k / float(n_draws)
    assert result["acceptance_rate"] == expected_rate

    # Independent check: every accepted sample's simulated summary is
    # within Euclidean distance eps of obs.
    if k > 0:
        # Re-simulate each accepted theta and verify ||s - obs||_2 <= eps.
        theta0_lo, theta0_hi = prior[0]
        for theta, d in zip(result["samples"], result["distances"]):
            assert len(theta) == 1
            assert theta0_lo <= theta[0] <= theta0_hi
            s = float(slope * theta[0] + intercept)
            # d is the distance between s and obs[0].
            expected_d = float(np.sqrt((s - float(obs[0])) ** 2))
            # Use a small tolerance for float round-trip through list->sum->sqrt.
            assert abs(d - expected_d) < 1e-12
            assert d <= eps + 1e-12

        # Independent check: posterior_mean is the arithmetic mean of samples.
        mean_theta0 = sum(t[0] for t in result["samples"]) / k
        assert abs(result["posterior_mean"][0] - mean_theta0) < 1e-12


def test_abcrej_edge():
    """Test edge case: very tight eps accepts nothing, posterior_mean is NaN."""
    rng = np.random.default_rng(1)
    obs = np.asarray([1000.0], dtype=float)  # far from any simulator value in [0,1]

    sim = _make_sim(slope=0.0, intercept=0.0)  # sim(theta, rng) = 0 always
    prior = [(0.0, 1.0)]
    eps = 1e-9  # tighter than |0 - 1000|
    n_draws = 50
    seed = 7

    result = abc_rejection(sim, obs, eps, prior, n_draws=n_draws, seed=seed)

    assert isinstance(result, dict)
    assert result["n_accepted"] == 0
    assert result["acceptance_rate"] == 0.0
    assert result["samples"] == []
    assert result["distances"] == []
    # With no accepted draws, posterior_mean is NaN per the implementation.
    assert len(result["posterior_mean"]) == len(prior)
    for v in result["posterior_mean"]:
        assert np.isnan(v)
