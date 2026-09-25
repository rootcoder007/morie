"""Tests for morie.fn.kbwpi — Sheather-Jones plug-in bandwidth."""

from morie.fn import _array_core as np
import pytest

from morie.fn.kbwpi import kbwpi


class TestKbwpi:
    def test_returns_positive(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = kbwpi(data)
        assert res["bw_opt"] > 0

    def test_normal_data_reasonable(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500)
        res = kbwpi(data)
        assert 0.05 < res["bw_opt"] < 1.0

    def test_n_returned(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = kbwpi(data)
        assert res["n"] == 5

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kbwpi(np.array([1.0]))


def test_kbwpi_matches_R_bw_SJ():
    """Equals R stats::bw.SJ on the same data: method "ste" (solved to
    tol = 1e-13 in R) and "dpi"."""
    import math
    x = [math.sin(1.7 * i) * 2 + 0.3 * math.cos(0.37 * i) for i in range(200)]
    assert kbwpi(x)["bw_opt"] == pytest.approx(0.27073444708748634, rel=1e-9)
    assert kbwpi(x, method="dpi")["bw_opt"] == pytest.approx(0.31627279834784933, rel=1e-12)
