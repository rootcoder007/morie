"""Tests for gh_c7_1.ghosal_pt_kl_prop."""

from morie.fn import _array_core as np

from morie.fn.gh_c7_1 import ghosal_pt_kl_prop


def test_gh_c7_1_basic():
    """Test basic functionality."""
    a_exponent = 2.0
    m_max = 200
    result = ghosal_pt_kl_prop(a_exponent=a_exponent, m_max=m_max)
    assert "estimate" in result
    assert "variance_series" in result
    assert "kl_property" in result
    est = float(np.asarray(result["estimate"], dtype=float))
    var = float(np.asarray(result["variance_series"], dtype=float))
    assert np.isfinite(est)
    assert np.isfinite(var)
    # Independent recomputation from the documented formula.
    expected_s_inv = sum(1.0 / float(m) ** a_exponent
                         for m in range(1, m_max + 1))
    expected_s_var = sum(1.0 / (4.0 * (2.0 * float(m) ** a_exponent + 1.0))
                         for m in range(1, m_max + 1))
    assert est == expected_s_inv
    assert var == expected_s_var
    # KL property holds iff the a_m series is summable, i.e. a_exponent > 1.
    assert bool(result["kl_property"]) == (a_exponent > 1.0)


def test_gh_c7_1_edge():
    """Test edge cases: minimal m_max and boundary a_exponent."""
    # Minimal series length: a single term.
    r1 = ghosal_pt_kl_prop(a_exponent=2.0, m_max=1)
    est1 = float(np.asarray(r1["estimate"], dtype=float))
    var1 = float(np.asarray(r1["variance_series"], dtype=float))
    assert est1 == 1.0 / (1.0 ** 2.0)
    assert var1 == 1.0 / (4.0 * (2.0 * (1.0 ** 2.0) + 1.0))
    assert bool(r1["kl_property"]) is True

    # a_exponent = 1 is the divergence boundary: sum 1/m diverges.
    r2 = ghosal_pt_kl_prop(a_exponent=1.0, m_max=1000)
    assert bool(r2["kl_property"]) is False
    # For a_exponent = 1, the estimate at m_max = 1000 should exceed the
    # Euler-Mascheroni-ish partial-sum bound ~ ln(1000) + gamma.
    est2 = float(np.asarray(r2["estimate"], dtype=float))
    assert est2 > float(np.log(1000.0))

    # a_exponent < 1: divergent, kl_property must be False.
    r3 = ghosal_pt_kl_prop(a_exponent=0.5, m_max=10)
    assert bool(r3["kl_property"]) is False
