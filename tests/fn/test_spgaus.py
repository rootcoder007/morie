"""Tests for spgaus.schabenberger_gaussian_variogram.

The book-certified identities for the whole variogram family live in
``test_schab_variogram_family.py``. This file pins the module's own
contract: it returns a real semivariogram, not the placeholder payload
it used to return.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spgaus import schabenberger_gaussian_variogram


def test_spgaus_returns_a_semivariogram():
    h = np.array([0.0, 0.5, 1.0, 2.0])
    r = schabenberger_gaussian_variogram(h, nugget=0.1, sill=1.0, range=1.0)
    g = r["gamma"]
    assert g[0] == 0.0
    assert np.all(np.diff(g) > 0)
    assert np.all(g <= 1.1 + 1e-12)
    assert r["model"] == "gaussian"


def test_spgaus_rejects_bad_input():
    with pytest.raises(ValueError):
        schabenberger_gaussian_variogram(np.array([1.0]), 0.0, 1.0, -1.0)
