"""Tests for diffRC.diffusion_rec."""

from morie.fn import _array_core as np

from morie.fn.diffRC import diffusion_rec


def test_diffRC_basic():
    """Test basic functionality."""
    rng_R = np.random.default_rng(42)
    rng_T = np.random.default_rng(43)
    R = rng_R.normal(0, 1, 100)
    T = rng_T.integers(0, 2, 100)

    n_steps = 10
    beta = [0.1] * n_steps
    # alpha_bar[t] = prod_{i=0..t} (1 - beta[i])
    alpha_bar = []
    prod = 1.0
    for b in beta:
        prod *= (1.0 - b)
        alpha_bar.append(prod)
    schedule = {
        "alpha_bar": alpha_bar,
        "beta": beta,
        "T": n_steps,
        "signal_retained": alpha_bar[-1],
    }

    # model: trivial denoiser returning the input unchanged
    def model(x, t):
        return list(x)

    result = diffusion_rec(R, model, schedule)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "x0" in result
    assert "path" in result
    assert "steps" in result
    assert result["steps"] == n_steps
    assert len(result["path"]) == n_steps
    assert len(result["estimate"]) == 100
    assert result["signal_retained"] == alpha_bar[-1]
    assert result["method"].startswith("diffusion")


def test_diffRC_edge():
    """Test edge cases."""
    rng_R = np.random.default_rng(42)
    rng_T = np.random.default_rng(43)
    R = rng_R.normal(0, 1, 100)
    T = rng_T.integers(0, 2, 100)

    n_steps = 5
    beta = [0.05] * n_steps
    alpha_bar = []
    prod = 1.0
    for b in beta:
        prod *= (1.0 - b)
        alpha_bar.append(prod)
    schedule = {
        "alpha_bar": alpha_bar,
        "beta": beta,
        "T": n_steps,
        "signal_retained": alpha_bar[-1],
    }

    def model(x, t):
        return list(x)

    result = diffusion_rec(R, model, schedule, t_start=2)
    assert isinstance(result, dict)
    assert result["steps"] == 3  # t = 2,1,0
    assert len(result["path"]) == 3
