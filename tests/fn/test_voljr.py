"""Tests for voljr."""

from morie.fn import _array_core as np
import pytest

from morie.fn.voljr import vol_jump_robust_var


def test_voljr_basic():
    rng = np.random.default_rng(42)
    r = rng.normal(scale=0.01, size=200)
    r[50] = 0.4
    out = vol_jump_robust_var(r)
    assert out["n_excluded"] >= 1
    assert out["rv"] - out["rv_truncated"] > 0.15


def test_voljr_edge():
    with pytest.raises(ValueError):
        vol_jump_robust_var([0.1, 0.2])
    with pytest.raises(ValueError):
        vol_jump_robust_var(np.ones(10), threshold=0.0)
