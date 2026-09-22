"""Tests for gh_c5_1.ghosal_dpm_model."""

from morie.fn import _array_core as np

from morie.fn.gh_c5_1 import ghosal_dpm_model


def test_gh_c5_1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dpm_model(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c5_1_edge():
    """Test edge cases."""
    x = np.array([42.0])
    result = ghosal_dpm_model(x)
    # The function returns a RichResult whose payload contains "density"
    # (one entry per input x). For a single-point input there must be
    # exactly one density value, and it must equal the scalar "estimate".
    assert "density" in result
    density = np.asarray(result["density"], dtype=float)
    assert density.shape == (1,)
    assert float(density[0]) == float(result["estimate"])
    # Independent computation of the documented normal mixture:
    # p_F(x) = sum_j W_j * N(x; theta_j, kernel_sd).
    # Using the same seed, alpha, n_terms, kernel_sd as defaults, we
    # can recompute the density from first principles via the stick-
    # breaking construction (independent of the function under test).
    import math
    rng = np.random.default_rng(42)
    M = float(1.0)  # alpha default
    n_terms = 200
    kernel_sd = 0.25
    V = [float(rng.beta(1.0, M)) for _ in range(n_terms)]
    # Stick-breaking: W_1 = V_1; W_j = V_j * prod_{k<j} (1 - V_k)
    W = []
    prod = 1.0
    for v in V:
        W.append(v * prod)
        prod *= (1.0 - v)
    th = [float(v) for v in rng.uniform(0, 1, n_terms)._flat()]
    xi = 42.0
    inv_sd = 1.0 / (kernel_sd * math.sqrt(2.0 * math.pi))
    dens = sum(w * inv_sd * math.exp(-0.5 * ((xi - t) / kernel_sd) ** 2)
               for w, t in zip(W, th))
    assert math.isclose(float(result["estimate"]), dens, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(sum(W), float(result["mixing_mass"]), rel_tol=1e-12, abs_tol=1e-12)
    assert result["method"] == "DP mixture density (GvdV 2017 eq. 5.1)"
