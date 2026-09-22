"""Tests for gh_c13_9.ghosal_ntr_levy."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c13_9 import ghosal_ntr_levy


def test_gh_c13_9_basic():
    """Test basic functionality."""
    f_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    nu_masses = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    result = ghosal_ntr_levy(f_vals, nu_masses)

    # Documented return key
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # Independent computation of the literature formula:
    # E exp(-int f dM) = exp(- sum_j m_j (1 - exp(-f_j)))
    expected_exponent = sum(
        m * (1.0 - math.exp(-f)) for f, m in zip(f_vals.tolist(), nu_masses.tolist())
    )
    expected_val = math.exp(-expected_exponent)
    assert math.isclose(result["estimate"], expected_val, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["exponent"], expected_exponent, rel_tol=1e-12, abs_tol=1e-12)


def test_gh_c13_9_edge():
    """Test edge cases: single Dirac mass."""
    f_vals = np.array([42.0])
    nu_masses = np.array([2.5])
    result = ghosal_ntr_levy(f_vals, nu_masses)

    # Independent computation for the single-mass case
    expected_exponent = 2.5 * (1.0 - math.exp(-42.0))
    expected_val = math.exp(-expected_exponent)
    assert math.isclose(result["estimate"], expected_val, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["exponent"], expected_exponent, rel_tol=1e-12, abs_tol=1e-12)
