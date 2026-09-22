"""Tests for gh_c4_20.ghosal_mix_dp."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_20 import ghosal_mix_dp


def test_gh_c4_20_basic():
    """Test basic functionality with documented argument shapes."""
    # G0_A_by_xi: probability of A under each component's base measure G0_xi,
    # one value per mixture component.
    G0_A_by_xi = np.array([0.1, 0.5, 0.9])
    # alpha_by_xi: concentration parameters (one per component), must be > -1
    # for the documented formula G0(1-G0)/(1+alpha_xi).
    alpha_by_xi = np.array([2.0, 5.0, 1.0])
    # pi_weights: non-negative component weights; need not be pre-normalised.
    pi_weights = np.array([1.0, 2.0, 1.0])

    result = ghosal_mix_dp(G0_A_by_xi, alpha_by_xi, pi_weights)

    # The function must return a mapping containing the documented keys.
    for key in ("estimate", "variance", "var_within", "var_between", "method"):
        assert key in result

    # Normalised weights: w_i = pi_i / sum(pi).
    w = pi_weights / np.sum(pi_weights)
    mean = np.sum(w * G0_A_by_xi)
    within = np.sum(w * G0_A_by_xi * (1.0 - G0_A_by_xi) / (1.0 + alpha_by_xi))
    between = np.sum(w * (G0_A_by_xi - mean) ** 2)

    # Independent recomputation of the documented formula.
    assert np.isclose(result["estimate"], mean)
    assert np.isclose(result["variance"], within + between)
    assert np.isclose(result["var_within"], within)
    assert np.isclose(result["var_between"], between)
    assert isinstance(result["method"], str)
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c4_20_edge():
    """Test the edge case of a single mixture component."""
    # With one component, prior mean must equal that component's G0(A)
    # and prior variance must equal G0(A)(1-G0(A))/(1+alpha_1).
    G0_A_by_xi = np.array([0.42])
    alpha_by_xi = np.array([3.0])
    pi_weights = np.array([1.0])

    result = ghosal_mix_dp(G0_A_by_xi, alpha_by_xi, pi_weights)

    w = pi_weights / np.sum(pi_weights)
    mean = np.sum(w * G0_A_by_xi)
    within = np.sum(w * G0_A_by_xi * (1.0 - G0_A_by_xi) / (1.0 + alpha_by_xi))
    between = np.sum(w * (G0_A_by_xi - mean) ** 2)

    assert np.isclose(result["estimate"], mean)
    assert np.isclose(result["variance"], within + between)
    assert np.isclose(result["var_between"], 0.0)
