"""Tests for gh_c11_3.ghosal_gp_crt_thm."""

from morie.fn import _array_core as np

from morie.fn.gh_c11_3 import ghosal_gp_crt_thm


def test_gh_c11_3_basic():
    """Test basic functionality: BM-style rate (phi_exponent=2.0, n=10000)."""
    phi_exponent = 2.0
    n = 10000
    result = ghosal_gp_crt_thm(phi_exponent=phi_exponent, n=n)
    # estimate key is documented and must exist
    assert "estimate" in result
    # the returned estimate must be a finite scalar
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # the method descriptor matches the literature citation in the docstring
    assert result["method"] == "Gaussian rate equation (GvdV 2017 Thm 11.20)"
    # independently evaluate the documented formula n^{-1/(2+a)} for BM (a=2)
    expected = float(n) ** (-1.0 / (2.0 + phi_exponent))
    assert float(result["estimate"]) == expected
    # the stored rate_exponent must equal 1/(2+a)
    assert float(result["rate_exponent"]) == 1.0 / (2.0 + phi_exponent)
    # the rate equation phi(eps) = n*eps^2 is solved up to a tiny residual
    a = float(phi_exponent)
    eps_n = expected
    expected_gap = abs(eps_n ** (-a) - float(n) * eps_n ** 2) \
        / (float(n) * eps_n ** 2)
    assert float(result["balance_gap"]) == expected_gap


def test_gh_c11_3_edge():
    """Test edge cases: scalar phi_exponent and n, plus n=1 baseline."""
    # default invocation (phi_exponent=2.0, n=10000) returns a single float
    result = ghosal_gp_crt_thm()
    assert isinstance(float(result["estimate"]), float)
    assert float(result["estimate"]) == float(10000) ** (-1.0 / 4.0)

    # phi_exponent=0 -> eps_n = n^{-1/2}
    result0 = ghosal_gp_crt_thm(phi_exponent=0.0, n=10000)
    assert float(result0["estimate"]) == float(10000) ** (-1.0 / 2.0)
    assert float(result0["rate_exponent"]) == 1.0 / 2.0

    # n=1 must produce eps_n = 1
    result1 = ghosal_gp_crt_thm(phi_exponent=2.0, n=1)
    assert float(result1["estimate"]) == 1.0
