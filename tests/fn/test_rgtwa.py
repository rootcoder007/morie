"""Tests for rgtwa.rangayyan_twave_alternans."""

import math

import pytest

from morie.fn.bsaqrs import rangayyan_twave_alternans


def test_rgtwa_basic():
    """Alternation of +-(a/2) g_j on every other beat puts (a/2)^2 g_j^2 in
    the 0.5 cycles-per-beat bin of sample j, so without noise the reported
    voltage is (a/2) rms(g)."""
    g = [math.sin(math.pi * j / 19) for j in range(20)]
    base = [2.0 * v for v in g]
    beats = [[base[j] + (0.03 if i % 2 else -0.03) * g[j] for j in range(20)] for i in range(32)]
    r = rangayyan_twave_alternans(beats)
    rms = math.sqrt(sum(v * v for v in g) / 20)
    assert r["valt"] == pytest.approx(0.03 * rms, rel=1e-10)
    assert r["noisemean"] < 1e-25


def test_rgtwa_edge():
    """With beat-to-beat noise the k-score measures the alternans against
    the noise band; no alternans, no detection."""
    g = [math.sin(math.pi * j / 19) for j in range(20)]
    noise = [[0.004 * math.sin(2.3 * i + 0.7 * j) for j in range(20)] for i in range(64)]
    alt = [[2 * g[j] + (0.05 if i % 2 else -0.05) * g[j] + noise[i][j] for j in range(20)] for i in range(64)]
    flat = [[2 * g[j] + noise[i][j] for j in range(20)] for i in range(64)]
    assert rangayyan_twave_alternans(alt)["present"] is True
    assert rangayyan_twave_alternans(flat)["present"] is False
    with pytest.raises(ValueError, match="eight beats"):
        rangayyan_twave_alternans(alt[:6])


