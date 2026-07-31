"""Tests for spkfnn.schabenberger_cross_validation_kriging.

Book identities for the kriging family live in test_schab_kriging.py.
This pins the module's own contract.
"""

import numpy as np
import pytest

from morie.fn.spkfnn import schabenberger_cross_validation_kriging

CM = {"nugget": 0.0, "sill": 1.0, "range": 2.0, "model": "exponential"}


def _field(n=20):
    rng = np.random.default_rng(0)
    c = rng.random((n, 2)) * 5.0
    return c, np.sin(c[:, 0]) + np.cos(c[:, 1])


def test_spkfnn_returns_a_real_estimate():
    c, z = _field()
    r = schabenberger_cross_validation_kriging(c, z, CM)
    assert r["mspe"] > 0
    assert r["residuals"].size == 20
    assert r["standardised"].size == 20
    assert r["rmspe"] == pytest.approx(np.sqrt(r["mspe"]))


def test_spkfnn_rejects_bad_input():
    with pytest.raises(ValueError, match="at least 3 points"):
        schabenberger_cross_validation_kriging(np.zeros((2, 2)), np.zeros(2), CM)
