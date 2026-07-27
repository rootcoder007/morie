"""Tests for copgau."""

import numpy as np
import pytest

from morie.fn._copula import copula_cdf
from morie.fn.copgau import gaussian_copula

def test_copgau_basic():
    out = gaussian_copula(0.3, 0.7, 0.5)
    assert out["cdf"] == pytest.approx(copula_cdf("gaussian", 0.3, 0.7, 0.5))
    assert out["tau"] == pytest.approx(2 / np.pi * np.arcsin(0.5))


def test_copgau_edge():
    assert gaussian_copula(0.4, 1.0, 0.5)["cdf"] == pytest.approx(0.4, abs=1e-5)
    with pytest.raises(ValueError):
        gaussian_copula(0.3, 0.7, 1.5)
