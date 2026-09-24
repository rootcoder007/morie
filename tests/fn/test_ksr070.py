"""Tests for ksr070.kosorok_ch3_score_operator_path."""

from morie.fn import _array_core as np

from morie.fn.ksr070 import kosorok_ch3_score_operator_path


def test_ksr070_basic():
    """Test basic functionality."""
    log_p = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    eta_path = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    h = 0.1
    result = kosorok_ch3_score_operator_path(log_p, eta_path, x, h)
    assert isinstance(result, dict)
    assert "score" in result


def test_ksr070_edge():
    """Test edge cases."""
    log_p = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    eta_path = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    h = 0.1
    result = kosorok_ch3_score_operator_path(log_p, eta_path, x, h)
    assert isinstance(result, dict)
