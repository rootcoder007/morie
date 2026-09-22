"""Tests for ddimst.ddim_step."""

from morie.fn import _array_core as np

from morie.fn.ddimst import ddim_step


def test_ddimst_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x_t = rng.normal(0, 1, 100)
    eps_theta = rng.normal(0, 1, 100)
    t = 5
    eta = 0.0
    result = ddim_step(x_t, t, eps_theta, eta)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "x_prev" in result
    assert "x0_pred" in result
    assert "sigma" in result
    assert "alpha_bar_t" in result
    assert "alpha_bar_prev" in result
    assert "n" in result
    assert result["n"] == 100
    assert isinstance(result["estimate"], float)
    # eta = 0 -> deterministic, sigma must be exactly 0
    assert result["sigma"] == 0.0


def test_ddimst_edge():
    """Test edge cases with eta = 1 (DDPM ancestral limit) and known schedule."""
    rng = np.random.default_rng(42)
    x_t = rng.normal(0, 1, 100)
    eps_theta = rng.normal(0, 1, 100)
    t = 5
    eta = 1.0
    # use explicit alpha_bar overrides so the test is independent of the schedule
    at = 0.4
    ap = 0.6
    result = ddim_step(x_t, t, eps_theta, eta=eta,
                       alpha_bar_t=at, alpha_bar_prev=ap)
    assert isinstance(result, dict)
    assert result["alpha_bar_t"] == at
    assert result["alpha_bar_prev"] == ap
    assert result["n"] == 100

    # independent recomputation of sigma with eta = 1
    sigma_expected = (1.0 * ((1.0 - ap) / (1.0 - at)) ** 0.5
                      * max(1.0 - at / ap, 0.0) ** 0.5)
    assert abs(result["sigma"] - sigma_expected) < 1e-12

    # independently recompute x0_pred and x_prev from the documented formula
    x0_expected = [(float(x_t[i]) - (1.0 - at) ** 0.5 * float(eps_theta[i]))
                   / at ** 0.5 for i in range(100)]
    c_expected = max(1.0 - ap - result["sigma"] * result["sigma"], 0.0) ** 0.5
    xp_expected = [ap ** 0.5 * x0_expected[i] + c_expected * float(eps_theta[i])
                   for i in range(100)]
    est_expected = sum(xp_expected) / 100

    for i in range(100):
        assert abs(result["x0_pred"][i] - x0_expected[i]) < 1e-12
        assert abs(result["x_prev"][i] - xp_expected[i]) < 1e-12
    assert abs(result["estimate"] - est_expected) < 1e-12
