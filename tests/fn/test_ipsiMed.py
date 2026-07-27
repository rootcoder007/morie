"""Tests for ipsiMed."""

import numpy as np
import pytest

from morie.fn.ipsiMed import interventional_psi


def test_ipsiMed_basic():
    rng = np.random.default_rng(42)
    n = 3000
    c = rng.normal(size=n)
    x = (rng.random(n) < 0.5).astype(float)
    m = 0.8 * x + 0.4 * c + rng.normal(scale=0.6, size=n)
    y = 0.5 * x + 1.0 * m + 0.3 * c + rng.normal(scale=0.6, size=n)
    out = interventional_psi(y, x, m, c=c)
    assert out["overall"] == pytest.approx(out["ide"] + out["iie"])
    assert out["iie"] == pytest.approx(0.8, abs=0.2)


def test_ipsiMed_edge():
    z = np.zeros(200)
    with pytest.raises(ValueError):
        interventional_psi(z, z, z)  # one exposure arm only
    with pytest.raises(ValueError):
        interventional_psi(z, (np.arange(200) % 2).astype(float), z, n_draws=10)
