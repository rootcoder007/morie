"""Tests for gh_conc_func.ghosal_concentration_function."""

from morie.fn import _array_core as np

from morie.fn.gh_conc_func import ghosal_concentration_function


def test_gh_conc_func_basic():
    """Test basic functionality."""
    # Documented inputs are two scalars (decentering_norm2 and small_ball_exp).
    decentering_norm2 = 1.5
    small_ball_exp = 0.75
    result = ghosal_concentration_function(decentering_norm2, small_ball_exp)

    # The estimate key is documented and must be finite.
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # Independent recomputation of the documented formula:
    # phi = decentering_norm2 + small_ball_exp.
    expected_phi = float(decentering_norm2) + float(small_ball_exp)
    assert float(result["estimate"]) == expected_phi

    # Other documented keys should be present and consistent.
    assert result["decentering"] == float(decentering_norm2)
    assert result["small_ball"] == float(small_ball_exp)


def test_gh_conc_func_edge():
    """Test edge cases (zeros)."""
    decentering_norm2 = 0.0
    small_ball_exp = 0.0
    result = ghosal_concentration_function(decentering_norm2, small_ball_exp)

    assert float(result["estimate"]) == 0.0
    assert result["decentering"] == 0.0
    assert result["small_ball"] == 0.0
