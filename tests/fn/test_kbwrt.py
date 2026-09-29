"""Tests for morie.fn.kbwrt — Silverman's rule-of-thumb bandwidth."""

import pytest

from morie.fn import _array_core as np
from morie.fn.kbwrt import kbwrt


class TestKbwrt:
    def test_gaussian_normal_data(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        res = kbwrt(data)
        assert res["bw"] > 0
        assert res["kernel"] == "gaussian"

    def test_silverman_formula(self):
        data = np.arange(1, 101, dtype=float)
        res = kbwrt(data)
        sigma = np.std(data, ddof=1)
        iqr = np.subtract(*np.percentile(data, [75, 25]))
        s = min(sigma, iqr / 1.34)
        expected = 0.9 * s * 100 ** (-0.2)
        assert res["bw"] == pytest.approx(expected, rel=1e-12)

    def test_epanechnikov_wider(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        gauss = kbwrt(data, kernel="gaussian")
        epan = kbwrt(data, kernel="epanechnikov")
        assert epan["bw"] > gauss["bw"]

    def test_raises_unknown_kernel(self):
        with pytest.raises(ValueError, match="Unknown"):
            kbwrt(np.array([1.0, 2.0, 3.0]), kernel="bogus")

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kbwrt(np.array([1.0]))

    def test_kernel_rescaling_is_the_canonical_bandwidth_ratio(self):
        """delta_0(K) = (R(K) / mu_2(K)^2)^(1/5); the ratio to the Gaussian
        delta_0 rescales the bandwidth (Marron and Nolan 1988)."""
        import math

        data = np.array([3.1, -0.4, 2.2, 5.9, 1.7, 0.3, 4.4, 2.8, -1.2, 3.6, 7.5, 1.1])
        g = kbwrt(data)["bw"]
        d0_gauss = (1.0 / (2.0 * math.sqrt(math.pi))) ** 0.2
        for kern, r, mu2 in (("epanechnikov", 0.6, 0.2), ("biweight", 5 / 7, 1 / 7), ("triweight", 350 / 429, 1 / 9)):
            want = g * (r / mu2**2) ** 0.2 / d0_gauss
            assert kbwrt(data, kernel=kern)["bw"] == pytest.approx(want, rel=1e-12)
