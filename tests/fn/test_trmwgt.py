"""Tests for trmwgt.trim_weights."""

import math

import pytest

from morie.fn.trmwgt import trim_weights


W = [1.2, 0.8, 3.5, 1.0, 9.0, 2.2, 0.9, 1.1, 14.0, 1.6]


def _q7(v, q):
    s = sorted(v)
    h = (len(s) - 1) * q
    lo = int(math.floor(h))
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def test_trmwgt_basic():
    """The cut is the type-7 quantile (R's default, numpy's linear); weights
    above it are set to it and the rescaled weights restore the total."""
    r = trim_weights(W, 0.8)
    cut = _q7(W, 0.8)
    assert r["estimate"] == pytest.approx(cut, rel=1e-15)
    assert r["weights"] == [min(v, cut) for v in W]
    assert r["n_trimmed"] == sum(1 for v in W if v > cut) == 2
    assert sum(r["rescaled"]) == pytest.approx(sum(W), rel=1e-15)
    assert r["cv_after"] < r["cv_before"]


def test_trmwgt_edge():
    """q = 1 trims nothing; negative weights and q outside (0, 1] raise."""
    r = trim_weights(W, 1.0)
    assert r["n_trimmed"] == 0 and r["weights"] == W
    with pytest.raises(ValueError, match="non-negative"):
        trim_weights([1.0, -2.0])
    with pytest.raises(ValueError, match="quantile"):
        trim_weights(W, 0.0)


