"""Tests for km131.kamath_ch9_input_projector."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.km131 import kamath_ch9_input_projector


def test_km131_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p, q = 4, 3, 2
    F_X = rng.normal(0, 1, (n, p))
    W = rng.normal(0, 1, (p, q))
    result = kamath_ch9_input_projector(F_X, W)

    estimate = result["estimate"]
    prompts = result["prompts"]
    shape = result["shape"]
    n_out = result["n"]
    method = result["method"]

    assert isinstance(estimate, float)
    assert math.isfinite(estimate)

    assert len(prompts) == n
    for row in prompts:
        assert len(row) == q
        for v in row:
            assert math.isfinite(v)

    assert tuple(shape) == (n, q)
    assert n_out == n
    assert isinstance(method, str)


def test_km131_edge():
    """Test edge cases: missing in_align raises ValueError."""
    rng = np.random.default_rng(42)
    F_X = rng.normal(0, 1, (4, 3))
    with pytest.raises(ValueError):
        kamath_ch9_input_projector(F_X)
    with pytest.raises(ValueError):
        kamath_ch9_input_projector(F_X, None)
