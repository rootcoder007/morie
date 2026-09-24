"""Tests for km079.kamath_ch6_alignscore_total_loss."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.km079 import kamath_ch6_alignscore_total_loss


def test_km079_basic():
    """Test basic functionality with valid scalar inputs."""
    L_3way = 1.5
    L_bin = 0.5
    L_reg = 2.0
    lambdas = [0.5, 0.3, 0.2]
    result = kamath_ch6_alignscore_total_loss(L_3way, L_bin, L_reg, lambdas)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "contributions" in result
    assert "losses" in result
    assert "lambdas" in result
    assert math.isfinite(result["estimate"])
    assert len(result["contributions"]) == 3
    assert len(result["losses"]) == 3
    assert len(result["lambdas"]) == 3
    assert result["n"] == 3
    assert result["method"] == "AlignScore joint loss (Kamath Eq 6.3)"


def test_km079_edge():
    """Test that negative weights raise ValueError per docstring."""
    with pytest.raises(ValueError):
        kamath_ch6_alignscore_total_loss(1.0, 2.0, 3.0, [0.5, -0.1, 0.6])
