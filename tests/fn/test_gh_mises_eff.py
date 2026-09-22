"""Tests for gh_mises_eff.ghosal_mises_efficiency."""

from morie.fn import _array_core as np

from morie.fn.gh_mises_eff import ghosal_mises_efficiency


def test_gh_mises_eff_basic():
    """Test basic functionality."""
    result = ghosal_mises_efficiency()
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(estimate)
    # Independent recomputation of the documented first-order expansion
    # gap for the contamination path P_h = (1-h) P0 + h delta_x at the
    # two fixed evaluation points x in (0.2, 0.8).
    t_eval = 0.5
    h = 0.02
    P0_t = t_eval
    gaps = []
    for x in (0.2, 0.8):
        psi_h = (1.0 - h) * P0_t + h * (1.0 if x <= t_eval else 0.0)
        infl = (1.0 if x <= t_eval else 0.0) - P0_t
        first_order = P0_t + h * infl
        gaps.append(abs(psi_h - first_order))
    expected_estimate = max(gaps)
    assert abs(estimate - expected_estimate) < 1e-14
    # Expansion is exact: influence function is linear in (P - P0).
    assert result["expansion_exact"] is True
    # Influence at x=0.2 against P0 = Uniform[0,1]: I(x) = 1{x<=t} - t.
    expected_infl_02 = (1.0 if 0.2 <= t_eval else 0.0) - P0_t
    assert abs(float(result["influence_at_02"]) - expected_infl_02) < 1e-14


def test_gh_mises_eff_edge():
    """Test edge cases: defaults-only call returns the documented keys."""
    result = ghosal_mises_efficiency()
    assert "estimate" in result
    assert "expansion_exact" in result
    assert "influence_at_02" in result
    assert "method" in result
