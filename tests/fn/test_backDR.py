"""Tests for backDR.back_door."""

from morie.fn import _array_core as np
import pytest

from morie.fn.backDR import back_door


def _confounded(seed=42, n=8000):
    rng = np.random.default_rng(seed)
    c = (rng.random(n) < 0.5).astype(int)
    x = (rng.random(n) < 0.2 + 0.6 * c).astype(int)
    y = (rng.random(n) < 0.1 + 0.4 * x + 0.4 * c).astype(int)
    return x, y, c


def test_backDR_basic():
    x, y, c = _confounded()
    out = back_door(y, x, c)
    dist = out["distribution"]
    # adjusted contrast recovers the 0.4 structural effect
    assert dist[1][1] - dist[0][1] == pytest.approx(0.4, abs=0.05)
    # the unadjusted contrast is inflated by the confounder
    naive = y[x == 1].mean() - y[x == 0].mean()
    assert naive - 0.4 > 0.05


def test_backDR_edge():
    x, y, c = _confounded()
    out = back_door(y, x, c)
    for row in out["distribution"].values():
        assert sum(row.values()) == pytest.approx(1.0)
    assert out["incomplete_strata"] == []
    with pytest.raises(ValueError):
        back_door(y[:10], x, c)  # length mismatch
