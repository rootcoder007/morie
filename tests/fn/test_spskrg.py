"""Tests for spskrg.schabenberger_simple_kriging.

Book identities for the kriging family live in test_schab_kriging.py.
This pins the module's own contract.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spskrg import schabenberger_simple_kriging

CM = {"nugget": 0.0, "sill": 1.0, "range": 2.0, "model": "exponential"}


def _field(n=20):
    rng = np.random.default_rng(0)
    c = rng.random((n, 2)) * 5.0
    return c, np.sin(c[:, 0]) + np.cos(c[:, 1])


def test_spskrg_returns_a_real_estimate():
    c, z = _field()
    r = schabenberger_simple_kriging(c, z, np.array([[2.0, 2.0]]), CM)
    assert r["prediction"].size == 1
    assert r["variance"][0] >= 0
    assert r["weights"].shape == (20, 1)
    assert np.isfinite(r["prediction"][0])


def test_spskrg_rejects_bad_input():
    c, z = _field()
    with pytest.raises(ValueError, match="same number of rows"):
        schabenberger_simple_kriging(c, z[:-1], np.array([[1.0, 1.0]]), CM)
