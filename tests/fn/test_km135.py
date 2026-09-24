"""Tests for km135.kamath_ch9_clip_contrastive_total."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.km135 import kamath_ch9_clip_contrastive_total


def test_km135_basic():
    """Test basic functionality."""
    L_i2t = 0.25
    L_t2i = 0.75
    result = kamath_ch9_clip_contrastive_total(L_i2t, L_t2i)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["estimate"] == L_i2t + L_t2i
    assert result["L_i2t"] == L_i2t
    assert result["L_t2i"] == L_t2i


def test_km135_edge():
    """Test edge cases."""
    # Both losses zero: smallest valid input.
    result = kamath_ch9_clip_contrastive_total(0.0, 0.0)
    assert math.isfinite(result["estimate"])
    assert result["estimate"] == 0.0
    # A negative cross-entropy is rejected per the docstring.
    with pytest.raises(ValueError):
        kamath_ch9_clip_contrastive_total(-0.1, 0.5)
