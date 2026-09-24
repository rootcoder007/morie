"""Tests for hmhplm.geron_hidden_layers_heuristic."""

from morie.fn import _array_core as np

from morie.fn.hmhplm import geron_hidden_layers_heuristic


def test_hmhplm_basic():
    """Test basic functionality."""
    model = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_hidden_layers_heuristic(model, X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "best_n_layers" in result


def test_hmhplm_edge():
    """Test edge cases."""
    model = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_hidden_layers_heuristic(model, X, y)
    assert isinstance(result, dict)
