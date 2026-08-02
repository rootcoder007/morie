"""Tests for mivbnd.monotone_iv_bounds."""

from morie.fn import _array_core as np
import pytest

from morie.fn.mivbnd import monotone_iv_bounds


def test_mivbnd_basic():
    rng = np.random.default_rng(42)
    n = 4000
    z = rng.integers(0, 4, n).astype(float)
    d = (rng.random(n) < 0.2 + 0.2 * z).astype(float)
    y = np.clip(0.3 * z + 2.0 * d + rng.normal(scale=0.3, size=n), -3, 6)
    out = monotone_iv_bounds(y, d, z, y_min=-3, y_max=6)
    assert out["lower"] <= 2.0 <= out["upper"]
    w_lo, w_hi = out["worst_case"]
    assert out["width"] <= (w_hi - w_lo) + 1e-9  # never wider than Manski


def test_mivbnd_edge():
    with pytest.raises(ValueError):
        monotone_iv_bounds([1.0, 2.0], [1, 0], [0.0, 0.0])  # one level
    with pytest.raises(ValueError):
        monotone_iv_bounds([1.0, 2.0], [0.5, 0.0], [0.0, 1.0])  # non-binary D
