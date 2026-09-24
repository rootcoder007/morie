"""Tests for chzlt.cinelli_hazlett."""

from morie.fn import _array_core as np
import math
import pytest

from morie.fn.chzlt import cinelli_hazlett


def test_chzlt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    model = rng.normal(0, 1, n)
    treat = rng.normal(0, 1, n)
    cov = rng.normal(0, 1, (n, p))
    R2_yu = 0.1
    R2_du = 0.2
    q = 0.5
    result = cinelli_hazlett(model, treat, cov, R2_yu, R2_du, q)
    assert isinstance(result, dict)
    expected_keys = {"estimate", "tau", "se", "t", "df", "bias", "adjusted_se",
                     "rv_q", "r2_yd_x", "robust", "n", "method"}
    assert expected_keys.issubset(result.keys())
    for key in ("estimate", "tau", "se", "t", "df", "bias", "adjusted_se",
                "rv_q", "r2_yd_x"):
        assert math.isfinite(result[key]), f"{key} not finite"
    assert result["robust"] in (0, 1)
    assert result["n"] == n


def test_chzlt_edge():
    """Test that invalid R2_yu raises ValueError."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    model = rng.normal(0, 1, n)
    treat = rng.normal(0, 1, n)
    cov = rng.normal(0, 1, (n, p))
    with pytest.raises(ValueError):
        cinelli_hazlett(model, treat, cov, R2_yu=1.5, R2_du=0.2)
