"""Tests for gxe_multitrait_model.gxe_multitrait_model."""

from morie.fn import _array_core as np

from morie.fn.gxe_multitrait_model import gxe_multitrait_model


def _make_inputs(n_lines, n_envs, n_traits, seed):
    """Build small plausible inputs for a multi-trait GxE LMM."""
    rng = np.random.default_rng(seed)
    n_obs = n_lines * n_envs

    Y = rng.normal(0, 1, (n_obs, n_traits))

    Z_L = [
        [1.0 if col == obs % n_lines else 0.0 for col in range(n_lines)]
        for obs in range(n_obs)
    ]

    Z_EL = [
        [1.0 if k == obs else 0.0 for k in range(n_lines * n_envs)]
        for obs in range(n_obs)
    ]

    G = np.eye(n_lines)
    Sigma_T = np.eye(n_traits)
    Sigma_E = np.eye(n_envs)
    Sigma_2T = np.eye(n_traits)
    R_T = np.eye(n_traits)

    return Y, Z_L, Z_EL, G, Sigma_T, Sigma_E, Sigma_2T, R_T


def test_msm032_basic():
    """Test basic functionality."""
    Y, Z_L, Z_EL, G, Sigma_T, Sigma_E, Sigma_2T, R_T = _make_inputs(
        n_lines=4, n_envs=2, n_traits=2, seed=42,
    )
    result = gxe_multitrait_model(
        Y, Z_L, Z_EL, G, Sigma_T, Sigma_E, Sigma_2T, R_T,
    )
    assert isinstance(result, dict)
    for key in ("estimate", "mu", "b_lines", "b_gxe", "method"):
        assert key in result


def test_msm032_edge():
    """Test edge case: single trait, single environment."""
    Y, Z_L, Z_EL, G, Sigma_T, Sigma_E, Sigma_2T, R_T = _make_inputs(
        n_lines=3, n_envs=1, n_traits=1, seed=7,
    )
    result = gxe_multitrait_model(
        Y, Z_L, Z_EL, G, Sigma_T, Sigma_E, Sigma_2T, R_T,
    )
    assert isinstance(result, dict)
    for key in ("estimate", "mu", "b_lines", "b_gxe", "method"):
        assert key in result
