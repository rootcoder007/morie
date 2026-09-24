"""Tests for rgcorec.rangayyan_correlation_coeff."""

import math

from morie.fn import _array_core as np
from morie.fn.bsastat import rangayyan_correlation_coeff


def test_rgcorec_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = rangayyan_correlation_coeff(x, y)
    r = float(result["r"])
    assert math.isfinite(r)
    # Perfect linear relationship y = 2x + 1 gives correlation 1.0
    assert abs(r - 1.0) < 1e-10


def test_rgcorec_edge():
    """Test edge cases."""
    result = rangayyan_correlation_coeff(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result["n"] == 2
