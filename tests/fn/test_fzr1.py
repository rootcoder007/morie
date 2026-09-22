"""Tests for fzr1.fauzi_r1_integral."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.fzr1 import fauzi_r1_integral


def test_fzr1_basic():
    """Test basic functionality with the gaussian closed form."""
    result = fauzi_r1_integral("gaussian")
    assert "estimate" in result
    assert "kernel" in result
    assert "method" in result
    assert result["kernel"] == "gaussian"
    # Closed form value: 1/(2*sqrt(pi))
    expected = 1.0 / (2.0 * np.sqrt(np.pi))
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert abs(float(result["estimate"]) - expected) < 1e-12


def test_fzr1_edge():
    """Test edge cases with a custom callable kernel and default grid."""
    # Use a simple symmetric kernel: uniform on [-1, 1]
    def K(y):
        return np.where(np.abs(y) <= 1.0, 0.5, 0.0)

    result = fauzi_r1_integral(K)
    assert "estimate" in result
    assert "kernel" in result
    assert "method" in result
    assert result["kernel"] == "callable"
    est = float(result["estimate"])
    assert np.isfinite(est)
    # By symmetry of a symmetric kernel on a symmetric grid,
    # W(y) is odd around 0 and y*K(y) is odd, so the integral is well-defined.
    assert est > 0.0
