"""Tests for gh_ap_g2.ghosal_dir_moments."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_g2 import ghosal_dir_moments


def test_gh_ap_g2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    A = float(x[0] + x[1] + x[2] + x[3] + x[4])
    j, jp = 0, 1
    expected_mean = float(x[j]) / A
    expected_var = float(x[j]) * (A - float(x[j])) / (A * A * (A + 1.0))
    expected_cov = -float(x[j]) * float(x[jp]) / (A * A * (A + 1.0))
    result = ghosal_dir_moments(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert np.isclose(result["estimate"], expected_mean)
    assert np.isclose(result["variance"], expected_var)
    assert np.isclose(result["covariance"], expected_cov)
    assert result["method"] == "Dirichlet moments (GvdV 2017 App G)"


def test_gh_ap_g2_edge():
    """Test edge cases: alpha must have >= 2 elements since jp=1 by default."""
    x = np.array([42.0, 7.0])
    A = float(x[0] + x[1])
    j, jp = 0, 1
    expected_mean = float(x[j]) / A
    expected_var = float(x[j]) * (A - float(x[j])) / (A * A * (A + 1.0))
    expected_cov = -float(x[j]) * float(x[jp]) / (A * A * (A + 1.0))
    result = ghosal_dir_moments(x)
    assert "estimate" in result
    assert np.isclose(result["estimate"], expected_mean)
    assert np.isclose(result["variance"], expected_var)
    assert np.isclose(result["covariance"], expected_cov)
