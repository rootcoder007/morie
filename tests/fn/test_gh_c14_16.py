"""Tests for gh_c14_16.ghosal_ncrm_levy."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_16 import ghosal_ncrm_levy


def test_gh_c14_16_basic():
    """Test basic functionality."""
    f_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    nu_masses = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    u_atoms = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

    # Independent computation of the documented formula:
    #   estimate = exp(-sum_j m_j * (1 - exp(-f_j * u_j)))
    exponent_expected = sum(
        m * (1.0 - np.exp(-f * u))
        for f, m, u in zip(f_vals, nu_masses, u_atoms)
    )
    estimate_expected = np.exp(-exponent_expected)

    result = ghosal_ncrm_levy(f_vals, nu_masses, u_atoms)
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert np.isclose(estimate, estimate_expected, rtol=1e-12, atol=1e-12)
    assert "exponent" in result
    assert np.isclose(
        float(np.asarray(result["exponent"], dtype=float)),
        float(exponent_expected),
        rtol=1e-12,
        atol=1e-12,
    )


def test_gh_c14_16_edge():
    """Test edge cases: a single atom matches the analytic expression."""
    f_vals = np.array([42.0])
    nu_masses = np.array([0.7])
    u_atoms = np.array([0.25])

    # For a single atom the formula reduces to exp(-m * (1 - exp(-f*u)))
    exponent_expected = 0.7 * (1.0 - np.exp(-42.0 * 0.25))
    estimate_expected = np.exp(-exponent_expected)

    result = ghosal_ncrm_levy(f_vals, nu_masses, u_atoms)
    assert "estimate" in result
    assert np.isclose(
        float(np.asarray(result["estimate"], dtype=float)),
        float(estimate_expected),
        rtol=1e-12,
        atol=1e-12,
    )
    assert float(np.asarray(result["estimate"], dtype=float)) > 0.0
    assert float(np.asarray(result["estimate"], dtype=float)) < 1.0
