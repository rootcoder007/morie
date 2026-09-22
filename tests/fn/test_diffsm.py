"""Tests for diffsm.diffusion_score_matching."""

from morie.fn import _array_core as np

from morie.fn.diffsm import diffusion_score_matching


def test_diffsm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, d = 100, 4
    x = rng.normal(0, 1, (n, d))

    def s_theta(z):
        return np.asarray(z)

    sigma = 1.0
    result = diffusion_score_matching(x, s_theta, sigma=sigma, n_noise=8, seed=42)
    assert isinstance(result, dict)
    assert "objective" in result
    assert "sigma" in result
    assert "per_sample" in result
    assert "target_norm" in result
    assert result["sigma"] == sigma
    assert result["per_sample"].shape == (n,)
    assert result["target_norm"] == 1.0 / sigma ** 2

    # Compute the expected objective independently using the documented formula:
    # J = E_{eps} || s_theta(x + sigma*eps) + (sigma*eps)/sigma^2 ||^2
    # averaged over n_noise draws and over n samples.
    rng2 = np.random.default_rng(42)
    n_noise = 8
    sigma = 1.0
    per = np.zeros(n)
    for _ in range(n_noise):
        eps = rng2.normal(size=(n, d))
        xt = x + sigma * eps
        s = np.asarray(s_theta(xt))
        target = -(xt - x) / sigma ** 2
        per += ((s - target) ** 2).sum(axis=1)
    per /= n_noise
    expected_objective = float(per.mean())

    assert np.isclose(result["objective"], expected_objective)
    assert np.isfinite(result["objective"])


def test_diffsm_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, d = 50, 3
    x = rng.normal(0, 1, (n, d))

    def s_theta(z):
        # An overly large score: the target term grows like 1/sigma while
        # s_theta does not, so the objective must blow up as sigma -> 0.
        return 1000.0 * np.asarray(z)

    sigma = 1.0
    result = diffusion_score_matching(x, s_theta, sigma=sigma, n_noise=4, seed=0)
    assert isinstance(result, dict)
    assert result["objective"] > 0.0
    assert np.isfinite(result["objective"])
    assert result["per_sample"].shape == (n,)
