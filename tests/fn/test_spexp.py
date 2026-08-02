"""Tests for spexp.schabenberger_exponential_variogram.

The book-certified identities for the whole variogram family live in
``test_schab_variogram_family.py``. This file pins the module's own
contract: it returns a real semivariogram, not the placeholder payload
it used to return.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spexp import schabenberger_exponential_variogram


def test_spexp_returns_a_semivariogram():
    h = np.array([0.0, 0.5, 1.0, 2.0])
    r = schabenberger_exponential_variogram(h, nugget=0.1, sill=1.0, range=1.0)
    g = r["gamma"]
    assert g.shape == h.shape
    assert g[0] == 0.0                      # nugget is a jump AT the origin
    assert np.all(np.diff(g) > 0)           # monotone increasing
    assert np.all(g <= 0.1 + 1.0 + 1e-12)   # bounded by the total sill
    assert r["model"] == "exponential"


def test_spexp_rejects_bad_input():
    with pytest.raises(ValueError):
        schabenberger_exponential_variogram(np.array([-1.0]), 0.0, 1.0, 1.0)
