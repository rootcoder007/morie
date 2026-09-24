"""Tests for facea.face_smooth."""

from morie.fn import _array_core as np
import math

from morie.fn.facea import face_smooth


def test_facea_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    n_curves = 40
    n_points = 50
    Y = rng.normal(0, 1, (n_curves, n_points))
    argvals = np.linspace(0.0, 1.0, n_points)
    result = face_smooth(Y, argvals)
    assert isinstance(result, dict)
    # check a real key that the function actually returns
    assert "covariance" in result
    # covariance should be a square matrix of size n_points
    cov = result["covariance"]
    assert len(cov) == n_points
    # eigenvalues should have one entry per grid point
    eig = result["eigenvalues"]
    assert len(eig) == n_points
    # noise_variance is a single number and must be finite
    assert math.isfinite(result["noise_variance"])


def test_facea_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    # minimum valid shapes: two curves and four grid points
    Y = rng.normal(0, 1, (2, 4))
    # keep n_basis small so the spline basis fits the short grid
    result = face_smooth(Y, n_basis=4)
    assert isinstance(result, dict)
    assert "covariance" in result
    cov = result["covariance"]
    assert len(cov) == 4
    eig = result["eigenvalues"]
    assert len(eig) == 4
