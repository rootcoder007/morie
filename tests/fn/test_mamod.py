"""Tests for morie.fn.mamod -- MA(q) model."""

import pytest

from morie.fn import _array_core as np
from morie.fn.mamod import ma_fit


class TestMAFit:
    def test_basic(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100)
        res = ma_fit(y, q=1)
        assert res.name == "ma_fit"
        assert len(res.extra["theta"]) == 1

    def test_short_raises(self):
        with pytest.raises(ValueError):
            ma_fit(np.ones(3), q=1)

    def test_cheatsheet(self):
        from morie.fn.mamod import cheatsheet

        assert isinstance(cheatsheet(), str)


def test_innovations_ma1_recomputed():
    """q = 1: theta = gamma_1 / gamma_0, v_1 = gamma_0 - theta^2 gamma_0."""
    y = [0.8, 1.1, 0.3, -0.2, 0.5, 1.4, 0.9, -0.1, 0.2, 0.7]
    n = 10
    m = sum(y) / n
    g0 = sum((v - m) ** 2 for v in y) / n
    g1 = sum((y[t] - m) * (y[t + 1] - m) for t in range(n - 1)) / n
    th = g1 / g0
    res = ma_fit(y, q=1)
    assert res.extra["theta"] == pytest.approx([th], rel=1e-12)
    assert res.extra["sigma2"] == pytest.approx(max(g0 - th * th * g0, 1e-10), rel=1e-12)
