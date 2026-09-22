"""Tests for gh_ap_j1.ghosal_levy_ito."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_j1 import ghosal_levy_ito


def test_gh_ap_j1_basic():
    """Test basic functionality."""
    fixed_atoms = (0.5,)
    atom_masses = (0.2,)
    result = ghosal_levy_ito(fixed_atoms=fixed_atoms, atom_masses=atom_masses,
                             n_random_jumps=200, seed=42)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # Independent computation of the formula:
    # total = sum(atom_masses) + sum_{i=1}^{n_random_jumps} Gamma(1/n_random_jumps, 1)
    fixed = sum(float(m) for m in atom_masses)
    rng = np.random.default_rng(42)
    random_part = sum(float(rng.gamma(1.0 / 200, 1.0)) for _ in range(200))
    expected_total = fixed + random_part
    assert np.isclose(float(result["estimate"]), expected_total)
    assert np.isclose(float(result["fixed_mass"]), fixed)
    assert np.isclose(float(result["poisson_mass"]), random_part)


def test_gh_ap_j1_edge():
    """Test edge cases: single fixed atom, no random jumps."""
    fixed_atoms = (0.5,)
    atom_masses = (42.0,)
    result = ghosal_levy_ito(fixed_atoms=fixed_atoms, atom_masses=atom_masses,
                             n_random_jumps=1, seed=42)
    # The function does not return key 'n'; correct keys are 'estimate',
    # 'fixed_mass', 'poisson_mass', 'method'.
    assert "estimate" in result
    assert "fixed_mass" in result
    assert "poisson_mass" in result
    assert "method" in result
    # fixed_mass should equal the sum of atom_masses.
    assert np.isclose(float(result["fixed_mass"]), 42.0)
    # With n_random_jumps=1, poisson_mass is a single Gamma(1, 1) draw.
    rng = np.random.default_rng(42)
    expected_poisson = float(rng.gamma(1.0 / 1, 1.0))
    assert np.isclose(float(result["poisson_mass"]), expected_poisson)
    assert np.isclose(float(result["estimate"]), 42.0 + expected_poisson)
