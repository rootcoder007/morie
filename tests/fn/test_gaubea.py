"""Tests for morie.fn.gaubea -- Gaussian beam optics."""

import numpy as np

from morie.fn._containers import DescriptiveResult
from morie.fn.gaubea import gaubea, gaussian_beam


class TestGaubea:
    def test_alias(self):
        assert gaubea is gaussian_beam

    def test_waist(self):
        r = gaussian_beam(wavelength=633e-9, w0=1e-3, z=[0])
        assert isinstance(r, DescriptiveResult)
        assert np.isclose(r.value["w"][0], 1e-3)

    def test_divergence(self):
        r = gaussian_beam(wavelength=633e-9, w0=1e-3, z=[0, 1, 2])
        assert r.value["w"][-1] > r.value["w"][0]
        assert r.value["z_rayleigh"] > 0
