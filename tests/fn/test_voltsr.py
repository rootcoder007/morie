"""Tests for voltsr."""

from morie.fn import _array_core as np
import pytest

from morie.fn.voltsr import vol_two_scale_rv


def test_voltsr_basic():
    rng = np.random.default_rng(42)
    r = rng.normal(scale=0.01, size=500)
    out = vol_two_scale_rv(r, K=5)
    assert out["tsrv"] == pytest.approx(0.01**2 * 500, rel=0.35)


def test_voltsr_edge():
    with pytest.raises(ValueError):
        vol_two_scale_rv(np.ones(6), K=5)
    with pytest.raises(ValueError):
        vol_two_scale_rv(np.ones(30), K=1)
