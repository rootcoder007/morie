"""Tests for hier.hierarchical_rl."""

import math

from morie.fn import _array_core as np
from morie.fn.hier import hierarchical_rl


def test_hier_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    env = rng.normal(0, 1, 100)
    options = rng.uniform(0, 1, 100)
    meta = 0.0
    result = hierarchical_rl(env, options, meta)
    assert isinstance(result, dict)
    # Check that all payload keys are present
    for key in ("estimate", "r_option", "k", "target", "td_error", "method"):
        assert key in result
    # Check numeric properties
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["r_option"])
    assert math.isfinite(result["target"])
    assert math.isfinite(result["td_error"])
    assert isinstance(result["k"], int)


def test_hier_edge():
    """Test edge cases."""
    rng = np.random.default_rng(123)
    env = rng.normal(0, 1, 1)  # single reward, no options
    meta = 0.5
    result = hierarchical_rl(env, meta=meta)
    assert isinstance(result, dict)
    assert result["k"] == 1
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
