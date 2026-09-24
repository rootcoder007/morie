"""Tests for ksr046.kosorok_ch2_z_estimator_consistency."""

from morie.fn import _array_core as np

from morie.fn.ksr046 import kosorok_ch2_z_estimator_consistency


def test_ksr046_basic():
    """Test basic functionality."""
    psi_n = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    psi = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    theta_seq = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    theta0 = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kosorok_ch2_z_estimator_consistency(psi_n, psi, theta_seq, theta0)
    assert isinstance(result, dict)
    assert "sup_differences" in result


def test_ksr046_edge():
    """Test edge cases."""
    psi_n = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    psi = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    theta_seq = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    theta0 = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kosorok_ch2_z_estimator_consistency(psi_n, psi, theta_seq, theta0)
    assert isinstance(result, dict)
