"""Tests for hrztpar.horowitz_parametric_T."""

import math
import pytest
from morie.fn import _array_core as np
from morie.fn.hrztpar import horowitz_parametric_T


def test_hrztpar_basic():
    """Test basic functionality with Box-Cox family."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    x = rng.normal(0, 1, (n, p))
    y = rng.uniform(0.5, 10.0, n)
    result = horowitz_parametric_T(x, y, T_family="boxcox")
    assert isinstance(result, dict)
    assert "theta_hat" in result
    assert "beta_hat" in result
    assert "criterion" in result
    assert math.isfinite(result["theta_hat"])
    assert math.isfinite(result["criterion"])
    assert -2.0 <= result["theta_hat"] <= 2.0
    assert len(result["beta_hat"]) == p


def test_hrztpar_edge():
    """Test with Bickel-Doksum family."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    x = rng.normal(0, 1, (n, p))
    y = rng.uniform(0.5, 10.0, n)
    # Bickel-Doksum requires a > 0, so restrict the search range to positive values
    result = horowitz_parametric_T(x, y, T_family="bickel-doksum", a_lo=0.1, a_hi=2.0)
    assert isinstance(result, dict)
    assert "theta_hat" in result
    assert "beta_hat" in result
    assert "criterion" in result
    assert math.isfinite(result["theta_hat"])
    assert math.isfinite(result["criterion"])
    assert 0.1 <= result["theta_hat"] <= 2.0
    assert len(result["beta_hat"]) == p
