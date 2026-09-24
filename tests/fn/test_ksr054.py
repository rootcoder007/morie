"""Tests for ksr054.kosorok_ch2_m_estimator_lipschitz_envelope."""

from morie.fn import _array_core as np

from morie.fn.ksr054 import kosorok_ch2_m_estimator_lipschitz_envelope


def test_ksr054_basic():
    """Test basic functionality."""
    m = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    m_dot = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    thetas = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kosorok_ch2_m_estimator_lipschitz_envelope(m, m_dot, thetas, x)
    assert isinstance(result, dict)
    assert "worst_ratio" in result


def test_ksr054_edge():
    """Test edge cases."""
    m = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    m_dot = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    thetas = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kosorok_ch2_m_estimator_lipschitz_envelope(m, m_dot, thetas, x)
    assert isinstance(result, dict)
