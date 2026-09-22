"""Tests for gh_c6_13.ghosal_lecam_consist."""

from morie.fn import _array_core as np

from morie.fn.gh_c6_13 import ghosal_lecam_consist


def test_gh_c6_13_basic():
    """Test basic functionality with all four required ingredients of the bound."""
    tv_P0_PU = 0.1          # total variation between P0 and P_U
    P0_phi = 0.2            # P0 phi
    Pi_U = 0.5              # Pi(U) (positive, used as denominator)
    int_V_P_1mphi = 0.3     # integral over V of P(1 - phi) dPi

    result = ghosal_lecam_consist(tv_P0_PU, P0_phi, Pi_U, int_V_P_1mphi)

    # The documented key must be present.
    assert "estimate" in result

    # Independent recomputation of the bound from the documented formula:
    #   P0 Pi(V|X) <= d_TV(P0, P_U) + P0 phi + (1/Pi(U)) int_V P(1-phi) dPi
    expected = float(tv_P0_PU) + float(P0_phi) \
        + float(int_V_P_1mphi) / float(Pi_U)
    assert np.isclose(float(result["estimate"]), expected)

    # Sanity: the estimate must be a finite, real-valued number.
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c6_13_edge():
    """Test edge case: scalar inputs still produce the documented 'estimate' key."""
    result = ghosal_lecam_consist(
        np.array(42.0),  # tv_P0_PU
        np.array(0.0),   # P0_phi
        np.array(1.0),   # Pi_U
        np.array(0.0),   # int_V_P_1mphi
    )
    assert "estimate" in result
    assert np.isclose(float(result["estimate"]), 42.0)
