"""Tests for sgtsbnd.sgt_sbm_detect_threshold."""

import math

from morie.fn.sgtsbnd import sgt_sbm_detect_threshold


def test_sgtsbnd_basic():
    """Strongly assortative parameters are above the Kesten-Stigum bound."""
    a, b, k = 10.0, 1.0, 2
    r = sgt_sbm_detect_threshold(a, b, k)
    c = (a + (k - 1) * b) / k
    thr = k * math.sqrt(c)
    assert abs(r["c"] - c) < 1e-9
    assert abs(r["threshold"] - thr) < 1e-9
    assert abs(r["margin"] - (abs(a - b) - thr)) < 1e-9
    assert r["detectable"] == 1.0
    assert r["estimate"] == r["detectable"]
    assert r["k"] == k


def test_sgtsbnd_undetectable():
    """a == b is the Erdos-Renyi case: no partition is detectable."""
    r = sgt_sbm_detect_threshold(0.5, 0.5)
    assert r["detectable"] == 0.0
    assert abs(r["c"] - 0.5) < 1e-9
    assert abs(r["threshold"] - 2.0 * math.sqrt(0.5)) < 1e-9
    assert abs(r["margin"] + 2.0 * math.sqrt(0.5)) < 1e-9


def test_sgtsbnd_edge():
    """k > 2 uses k sqrt(c), not the stub's extra 1/(k-1) factor."""
    a, b, k = 20.0, 2.0, 4
    r = sgt_sbm_detect_threshold(a, b, k)
    c = (a + (k - 1) * b) / k
    assert abs(r["c"] - 6.5) < 1e-9
    assert abs(r["threshold"] - k * math.sqrt(c)) < 1e-9
    assert r["detectable"] == (1.0 if abs(a - b) > k * math.sqrt(c) else 0.0)
