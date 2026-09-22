"""Tests for gh_c3_8.ghosal_moment_prior."""

from morie.fn import _array_core as np

from morie.fn.gh_c3_8 import ghosal_moment_prior


def test_gh_c3_8_basic():
    """Test basic functionality with a feasible moment sequence (uniform on [0,1])."""
    # Uniform distribution on [0,1] has moments m_k = 1/(k+1)
    moments = [1.0, 1/2, 1/3, 1/4, 1/5]
    result = ghosal_moment_prior(moments)
    assert isinstance(result, dict)
    for key in ("feasible", "min_difference", "n_violations", "order", "differences"):
        assert key in result
    assert result["feasible"] == 1.0
    assert result["n_violations"] == 0.0
    assert result["order"] == 4.0  # len(moments) - 1
    assert result["min_difference"] >= -1e-12
    # Differences triangle: row k should have N-k entries
    assert len(result["differences"]) == 5


def test_gh_c3_8_edge():
    """Test edge cases: point mass at 0, single moment, infeasible sequence."""
    # Single moment case
    result_single = ghosal_moment_prior([1.0])
    assert result_single["feasible"] == 1.0
    assert result_single["order"] == 0.0
    assert result_single["min_difference"] == 1.0
    assert result_single["n_violations"] == 0.0

    # Point mass at 0: moments [1, 0, 0, 0, 0]
    result_pm = ghosal_moment_prior([1.0, 0.0, 0.0, 0.0, 0.0])
    assert result_pm["feasible"] == 1.0
    assert result_pm["n_violations"] == 0.0
    assert result_pm["min_difference"] >= -1e-12

    # Infeasible: m_1 = 2 means more mass than possible on [0,1]
    # Moments [1, 2, 2]: Delta^1 = [1, 0], (-1)*Delta^1 = [-1, 0] -> violation
    result_infeasible = ghosal_moment_prior([1.0, 2.0, 2.0])
    assert result_infeasible["feasible"] == 0.0
    assert result_infeasible["n_violations"] >= 1.0
    assert result_infeasible["min_difference"] < 0


def test_gh_c3_8_invalid():
    """Test that invalid input (m_0 != 1) raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="m_0"):
        ghosal_moment_prior([0.5, 0.3, 0.2])
