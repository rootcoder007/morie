"""Tests for ghs033.ghosal_ch3_polya_tree_mixture_second_kind."""

from morie.fn import _array_core as np

from morie.fn.ghs033 import ghosal_ch3_polya_tree_mixture_second_kind


def _alpha_path_factory(paths):
    """Build an alpha_path_of_theta callable mapping theta -> iterable of
    (alpha_taken, alpha_other) pairs, given one path per theta.
    """
    by_theta = {float(th): list(p) for th, p in paths}

    def alpha_path_of_theta(th, x):
        return by_theta[float(th)]

    return alpha_path_of_theta


def test_ghs033_basic():
    """Test basic functionality with a uniform mixture (weights=None)."""
    rng = np.random.default_rng(42)
    thetas = [0.0, 1.0]

    # For each theta, build a simple 2-level path with symmetric alphas so
    # 2*a / (a + a) == 1 at every level -> per_theta = 1, mixture = 1.
    paths = [
        (0.0, [(1.0, 1.0), (1.0, 1.0)]),
        (1.0, [(1.0, 1.0), (1.0, 1.0)]),
    ]
    alpha_path_of_theta = _alpha_path_factory(paths)
    x = float(rng.normal(0, 1))  # x is scalar per the docstring

    result = ghosal_ch3_polya_tree_mixture_second_kind(
        x, alpha_path_of_theta, thetas
    )
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "distribution" in result
    assert "per_theta" in result
    assert "method" in result

    # Independent expectation: per_theta = prod_j 2 a_j / (a_j + a_j) = 1
    # for every theta, weights default to uniform -> mix = sum(w_i * 1) = 1.
    expected_per = [1.0, 1.0]
    expected_mix = sum((1.0 / 2) * g for g in expected_per)
    assert np.allclose(result["per_theta"], expected_per)
    assert np.isclose(result["estimate"], expected_mix)
    assert np.isclose(result["distribution"], expected_mix)


def test_ghs033_edge():
    """Test edge cases: custom weights and asymmetric alpha path."""
    thetas = [0.0, 2.0]
    # Asymmetric path for theta=0.0; for theta=2.0 a uniform path -> 1.
    paths = [
        (0.0, [(2.0, 1.0), (3.0, 1.0)]),
        (2.0, [(1.0, 1.0), (1.0, 1.0)]),
    ]
    alpha_path_of_theta = _alpha_path_factory(paths)
    x = 0.5  # scalar x per the docstring

    weights = [0.25, 0.75]
    result = ghosal_ch3_polya_tree_mixture_second_kind(
        x, alpha_path_of_theta, thetas, weights=weights
    )
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "per_theta" in result

    # Independent computation of per_theta from the documented formula.
    g0 = (2.0 * 2.0 / (2.0 + 1.0)) * (2.0 * 3.0 / (3.0 + 1.0))
    g1 = 1.0
    expected_per = [g0, g1]
    expected_mix = 0.25 * g0 + 0.75 * g1
    assert np.allclose(result["per_theta"], expected_per)
    assert np.isclose(result["estimate"], expected_mix)
