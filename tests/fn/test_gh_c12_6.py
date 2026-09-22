"""Tests for gh_c12_6.ghosal_semipara_eff."""

from morie.fn import _array_core as np

from morie.fn.gh_c12_6 import ghosal_semipara_eff


def test_gh_c12_6_basic():
    """Test basic functionality."""
    grad_psi = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    info_matrix = np.array([
        [4.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 5.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 6.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 5.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 4.0],
    ])
    result = ghosal_semipara_eff(grad_psi, info_matrix)
    assert "estimate" in result
    # Independent computation of (grad psi)' I^{-1} (grad psi):
    info_inv = np.linalg.inv(np.asarray(info_matrix, dtype=float))
    expected = float(np.asarray(grad_psi, dtype=float) @ info_inv @ np.asarray(grad_psi, dtype=float))
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert np.isclose(result["estimate"], expected)
    assert result["positive"] == (expected > 0)
    assert result["method"] == "semiparametric efficiency bound (GvdV 2017 sec. 12.3)"


def test_gh_c12_6_edge():
    """Test edge cases."""
    grad_psi = np.array([42.0])
    info_matrix = np.array([[7.0]])
    result = ghosal_semipara_eff(grad_psi, info_matrix)
    assert "estimate" in result
    # (42)^2 / 7 = 1764 / 7 = 252
    assert np.isclose(result["estimate"], 42.0 * 42.0 / 7.0)
    assert result["positive"] is True
