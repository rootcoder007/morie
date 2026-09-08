"""Tests for spsph.schabenberger_spherical_variogram.

The book-certified identities for the whole variogram family live in
``test_schab_variogram_family.py``. This file pins the module's own
contract: it returns a real semivariogram, not the placeholder payload
it used to return.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spsph import schabenberger_spherical_variogram


def test_spsph_returns_a_semivariogram():
    h = np.array([0.0, 0.5, 1.0, 3.0])
    r = schabenberger_spherical_variogram(h, nugget=0.0, sill=2.0, range=1.0)
    g = r["gamma"]
    assert g[0] == 0.0
    assert g[2] == pytest.approx(2.0)       # true range: sill reached AT alpha
    assert g[3] == pytest.approx(2.0)       # and flat beyond it
    assert r["model"] == "spherical"


def test_spsph_rejects_bad_input():
    with pytest.raises(ValueError):
        schabenberger_spherical_variogram(np.array([1.0]), -1.0, 1.0, 1.0)
