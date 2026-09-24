"""Tests for ipfsfa.ipopt_solver."""

import math
import pytest

from morie.fn import _array_core as np
from morie.fn.ipfsfa import ipopt_solver


def test_ipfsfa_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x0 = rng.normal(0, 1, 5)
    f = lambda x: sum([xi * xi for xi in x])
    constraints = [lambda x: -1.0 - sum([xi * xi for xi in x])]
    result = ipopt_solver(f, constraints, x0, outer=2, inner=5)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])


def test_ipfsfa_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x0 = rng.normal(0, 1, 5)
    f = lambda x: sum([xi * xi for xi in x])
    constraints = [lambda x: 1.0 + sum([xi * xi for xi in x])]
    with pytest.raises(ValueError):
        ipopt_solver(f, constraints, x0, outer=1, inner=1)
