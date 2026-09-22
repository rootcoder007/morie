"""Tests for gh_c4_5.ghosal_dp_selfsim."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_5 import ghosal_dp_selfsim


def test_gh_c4_5_basic():
    """Test basic functionality."""
    w = 0.4
    base_masses_in_A = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    base_masses_in_Ac = np.array([0.5, 1.5, 2.5])

    result = ghosal_dp_selfsim(w, base_masses_in_A, base_masses_in_Ac)

    assert "estimate" in result
    estimate = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(estimate))

    # Documented keys the function returns
    assert "P_cells" in result
    assert "total_mass" in result
    assert "method" in result

    p_cells = np.asarray(result["P_cells"], dtype=float)
    total_mass = float(result["total_mass"])

    # Number of cells = len(A) + len(A^c)
    assert p_cells.shape == (base_masses_in_A.size + base_masses_in_Ac.size,)

    # Recompute expected weights from gamma draws with the documented seed
    rng_a = np.random.default_rng(42)
    gammas_a = [float(rng_a.gamma(max(a, 1e-12), 1.0)) for a in np.ravel(base_masses_in_A)]
    s_a = sum(gammas_a)
    pA_expected = [v / s_a for v in gammas_a]

    rng_b = np.random.default_rng(42)
    _ = [float(rng_b.gamma(max(a, 1e-12), 1.0)) for a in np.ravel(base_masses_in_A)]
    gammas_b = [float(rng_b.gamma(max(a, 1e-12), 1.0)) for a in np.ravel(base_masses_in_Ac)]
    s_b = sum(gammas_b)
    pAc_expected = [v / s_b for v in gammas_b]

    expected = [w * v for v in pA_expected] + [(1.0 - w) * v for v in pAc_expected]
    expected_total = sum(expected)
    expected_estimate = expected[0]

    assert np.allclose(p_cells, expected)
    assert np.isclose(total_mass, expected_total)
    assert np.isclose(float(estimate), expected_estimate)


def test_gh_c4_5_edge():
    """Test edge cases."""
    w = 0.3
    result = ghosal_dp_selfsim(w, np.array([42.0]), np.array([7.0]))
    # Documented behaviour: 'estimate' is the first recombined weight.
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(estimate)
