"""Tests for volyz."""

import numpy as np
import pytest

from morie.fn.volyz import vol_yang_zhang


def test_volyz_basic():
    rng = np.random.default_rng(42)
    n, steps = 200, 40
    sig_step = 0.02 / np.sqrt(steps)
    o = np.empty(n); h = np.empty(n); l = np.empty(n); c = np.empty(n)
    p = 0.0
    for d in range(n):
        o[d] = p
        path = p + np.cumsum(rng.normal(scale=sig_step, size=steps))
        h[d] = max(path.max(), p); l[d] = min(path.min(), p); c[d] = path[-1]
        p = c[d]
    out = vol_yang_zhang(np.exp(o), np.exp(h), np.exp(l), np.exp(c))
    assert out["sigma"] == pytest.approx(0.02, rel=0.25)


def test_volyz_edge():
    with pytest.raises(ValueError):
        vol_yang_zhang([100.0], [110.0], [95.0], [105.0])  # single day
