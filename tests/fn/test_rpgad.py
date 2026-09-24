"""Tests for rpgad.rdp_to_eps_delta."""
import math

import pytest

from morie.fn import _array_core as np
from morie.fn.rpgad import rdp_to_eps_delta


def test_rpgad_basic():
    """Test basic functionality with the Gaussian closed-form curve."""
    alpha = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    result = rdp_to_eps_delta(
        alpha,
        mechanism="gaussian",
        sigma=2.0,
        delta=1e-6,
        sensitivity=1.0,
        n_compositions=10,
    )
    assert isinstance(result, dict)
    eps = float(result["estimate"])
    assert math.isfinite(eps)
    assert eps >= 0.0
    assert "best_alpha" in result
    best = float(result["best_alpha"])
    assert any(abs(best - float(a)) < 1e-9 for a in alpha)


def test_rpgad_edge():
    """Test that inputs the docstring says are invalid raise ValueError."""
    with pytest.raises(ValueError):
        rdp_to_eps_delta(1.0, mechanism="gaussian", sigma=1.0)
    with pytest.raises(ValueError):
        rdp_to_eps_delta(2.0, mechanism="gaussian", sigma=1.0, delta=0.0)
    with pytest.raises(ValueError):
        rdp_to_eps_delta(2.0, mechanism="gaussian", sigma=1.0, delta=1.0)
