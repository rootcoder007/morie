"""Tests for tinfo.test_information (IRT test information)."""

import math

import pytest

from morie.fn.tinfo import test_information as tinfo_fn


def _p(t, a, b, c, u, D):
    return c + (u - c) / (1 + math.exp(-D * a * (t - b)))


def test_tinfo_basic():
    """I(theta) = sum_i P_i'^2 / (P_i Q_i).  P' is taken by a central
    difference with h = 1e-5: the truncation error is h^2 P'''/6 ~ 1e-11
    and the rounding error eps/h ~ 1e-11, so 1e-8 relative is safe."""
    a, b, c, u = [1.2, 0.8, 1.5], [-0.5, 0.3, 1.1], [0.2, 0.0, 0.1], [1.0, 0.95, 1.0]
    th = [-1.0, 0.4, 2.0]
    r = tinfo_fn(th, a, b, c, u, D=1.0)
    h = 1e-5
    for t, tot, per in zip(th, r["information"], r["item_information"]):
        ref = []
        for z in zip(a, b, c, u):
            p = _p(t, *z, 1.0)
            dp = (_p(t + h, *z, 1.0) - _p(t - h, *z, 1.0)) / (2 * h)
            ref.append(dp * dp / (p * (1 - p)))
        assert per == pytest.approx(ref, rel=1e-8)
        assert tot == pytest.approx(sum(ref), rel=1e-8)
    assert r["sem"] == pytest.approx([1 / math.sqrt(v) for v in r["information"]], rel=1e-15)


def test_tinfo_edge():
    """A 2PL item at theta = b gives D^2 a^2 / 4; guessing lowers it."""
    r = tinfo_fn(0.5, [2.0], [0.5], D=1.702)
    assert r["information"] == pytest.approx(1.702 ** 2 * 4 / 4, rel=1e-14)
    assert tinfo_fn(0.5, [2.0], [0.5], c=[0.2], D=1.702)["information"] < r["information"]
    with pytest.raises(ValueError):
        tinfo_fn(0.0, [1.0], [0.0], c=[1.0])
