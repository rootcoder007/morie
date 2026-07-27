"""Tests for msmiv2.msm_iv."""

import numpy as np
import pytest

from morie.fn.msmiv2 import msm_iv


def test_msmiv2_basic():
    rng = np.random.default_rng(42)
    n = 3000
    u = rng.normal(size=n)
    z = rng.normal(size=n)
    a = 0.7 * z + u + rng.normal(scale=0.5, size=n)
    y = 1.2 * a + 2.0 * u + rng.normal(scale=0.5, size=n)
    out = msm_iv(y, a, z)
    assert out["estimate"] == pytest.approx(1.2, abs=0.15)
    assert abs(out["ols_estimate"] - 1.2) > 0.3
    assert out["weak_instrument"] is False


def test_msmiv2_edge():
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.normal(size=n)
    y = rng.normal(size=n)
    assert msm_iv(y, a, rng.normal(size=n) * 0.001)["weak_instrument"] is True
    with pytest.raises(ValueError):
        msm_iv(y[:10], a, a)  # length mismatch
