"""Tests for spmatr.schabenberger_matern_covariance.

Book identities for the family live in test_schab_matern_family.py.
This pins the module's own contract.
"""

import numpy as np
import pytest

from morie.fn.spmatr import schabenberger_matern_covariance


def test_spmatr_returns_a_covariance():
    h = np.array([0.0, 0.5, 1.0, 3.0])
    r = schabenberger_matern_covariance(h, sigma2=2.0, nu=1.5, a=1.0)
    c = r["covariance"]
    assert c.shape == h.shape
    assert c[0] == pytest.approx(2.0)                 # C(0) is the variance
    assert np.all(np.diff(c) < 0)                     # decays with lag
    np.testing.assert_allclose(r["semivariogram"], 2.0 - c, rtol=1e-12)
    assert r["nu"] == 1.5 and r["theta"] == 1.0


def test_spmatr_rejects_bad_input():
    with pytest.raises(ValueError, match="`nu` must be"):
        schabenberger_matern_covariance(np.array([1.0]), 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        schabenberger_matern_covariance(np.array([-1.0]), 1.0, 1.0, 1.0)
