"""Tests for morie.fn.armod -- AR(p) model."""

import pytest

from morie.fn import _array_core as np
from morie.fn.armod import ar_fit


class TestARFit:
    def test_basic_ar1(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(100)) * 0.1
        res = ar_fit(y, p=1)
        assert res.name == "ar_fit"
        assert len(res.extra["phi"]) == 1
        assert res.extra["n"] == 100

    def test_ar2(self):
        rng = np.random.default_rng(7)
        y = rng.standard_normal(200)
        res = ar_fit(y, p=2)
        assert len(res.extra["phi"]) == 2

    def test_short_raises(self):
        with pytest.raises(ValueError):
            ar_fit(np.ones(2), p=1)

    def test_cheatsheet(self):
        from morie.fn.armod import cheatsheet

        assert isinstance(cheatsheet(), str)

    def test_yule_walker_recomputed(self):
        """AR(2) Yule-Walker: solve [[r0, r1], [r1, r0]] phi = [r1, r2]."""
        import pytest

        y = [0.8, 1.1, 0.3, -0.2, 0.5, 1.4, 0.9, -0.1, 0.2, 0.7, 1.0, 0.4]
        n = len(y)
        m = sum(y) / n
        c = [sum((y[t] - m) * (y[t + k] - m) for t in range(n - k)) / n for k in range(3)]
        det = c[0] ** 2 - c[1] ** 2
        phi = [(c[1] * c[0] - c[1] * c[2]) / det, (c[0] * c[2] - c[1] * c[1]) / det]
        res = ar_fit(y, p=2)
        assert res.extra["phi"] == pytest.approx(phi, rel=1e-12)
        assert res.extra["sigma2"] == pytest.approx(c[0] - phi[0] * c[1] - phi[1] * c[2], rel=1e-12)
        assert res.extra["mean"] == pytest.approx(m, rel=1e-14)
