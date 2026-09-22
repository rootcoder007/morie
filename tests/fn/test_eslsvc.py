"""Tests for eslsvc.esl_svc."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.eslsvc import esl_svc


def test_eslsvc_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    y = rng.choice([-1.0, 1.0], size=100)
    result = esl_svc(X, y, C=1.0)
    assert isinstance(result, dict)
    # Core keys documented in the docstring
    for key in ("w", "b", "margin", "alpha", "support_",
                "decision", "class_", "accuracy", "n_violations"):
        assert key in result
    # margin is 2 / ||w||, computed independently
    w = np.asarray(result["w"], dtype=float)
    expected_margin = 2.0 / float(np.linalg.norm(w))
    assert abs(result["margin"] - expected_margin) < 1e-8
    # w should match alpha_i * y_i summed into X
    alpha = np.asarray(result["alpha"], dtype=float)
    ypm = np.where(y == np.unique(y)[1], 1.0, -1.0)
    expected_w = (alpha * ypm) @ np.asarray(X, dtype=float)
    assert np.allclose(np.asarray(result["w"], dtype=float), expected_w)
    # decision on training data equals X w + b
    Xf = np.asarray(X, dtype=float)
    expected_dec = Xf @ np.asarray(result["w"], dtype=float) + float(result["b"])
    assert np.allclose(np.asarray(result["decision"], dtype=float), expected_dec)
    # accuracy in [0, 1]
    assert 0.0 <= float(result["accuracy"]) <= 1.0


def test_eslsvc_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    y = rng.choice([-1.0, 1.0], size=100)
    result = esl_svc(X, y, C=1.0)
    assert isinstance(result, dict)
    assert "w" in result and "b" in result and "margin" in result
