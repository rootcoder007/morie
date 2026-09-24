"""Tests for km105.kamath_ch6_gedi_combined_loss."""

from morie.fn import _array_core as np

import math

from morie.fn.km105 import kamath_ch6_gedi_combined_loss


def test_km105_basic():
    """Test basic functionality with scalars matching the convex combination formula."""
    L_g = 1.0
    L_d = 3.0
    lam = 0.25
    result = kamath_ch6_gedi_combined_loss(L_g, L_d, lam)
    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    assert result["estimate"] == 2.5
    assert result["contributions"] == [0.25, 2.25]
    assert result["L_g"] == 1.0
    assert result["L_d"] == 3.0
    assert result["lam"] == 0.25


def test_km105_edge():
    """Test edge cases at the convex-combination boundaries lam=0 and lam=1."""
    # lam = 1: pure language model, estimate equals L_g exactly.
    r1 = kamath_ch6_gedi_combined_loss(2.0, 5.0, 1.0)
    assert isinstance(r1, dict)
    assert math.isfinite(r1["estimate"])
    assert r1["estimate"] == 2.0
    assert r1["contributions"] == [2.0, 0.0]

    # lam = 0: pure discriminator, estimate equals L_d exactly.
    r2 = kamath_ch6_gedi_combined_loss(2.0, 5.0, 0.0)
    assert isinstance(r2, dict)
    assert math.isfinite(r2["estimate"])
    assert r2["estimate"] == 5.0
    assert r2["contributions"] == [0.0, 5.0]
