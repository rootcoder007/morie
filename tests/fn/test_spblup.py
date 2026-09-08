"""Tests for spblup.schabenberger_blup.

Book identities for the kriging family live in test_schab_kriging.py.
This pins the module's own contract.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spblup import schabenberger_blup

CM = {"nugget": 0.0, "sill": 1.0, "range": 2.0, "model": "exponential"}


def _field(n=20):
    rng = np.random.default_rng(0)
    c = rng.random((n, 2)) * 5.0
    return c, np.sin(c[:, 0]) + np.cos(c[:, 1])


def test_spblup_returns_a_real_estimate():
    c, z = _field()
    r = schabenberger_blup(c, z, np.array([[2.0, 2.0]]), CM)
    assert r["weights"].sum(axis=0)[0] == pytest.approx(1.0, abs=1e-10)
    assert r["variance"][0] >= 0
    assert np.isfinite(r["lagrange"][0])


def test_spblup_rejects_bad_input():
    c, z = _field()
    with pytest.raises(ValueError, match="same number of rows"):
        schabenberger_blup(c, z[:-1], np.array([[1.0, 1.0]]), CM)
