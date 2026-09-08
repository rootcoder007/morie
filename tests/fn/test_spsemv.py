"""Tests for spsemv.schabenberger_semivariogram_def.

The book-certified identities for the whole variogram family live in
``test_schab_variogram_family.py``. This file pins the module's own
contract: it returns a real semivariogram, not the placeholder payload
it used to return.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spsemv import schabenberger_semivariogram_def


def test_spsemv_returns_a_semivariogram():
    rng = np.random.default_rng(4)
    coords = rng.random((200, 2))
    z = 4.0 * coords[:, 0] + rng.normal(0, 0.1, 200)
    r = schabenberger_semivariogram_def(coords, z, n_bins=5)
    assert r["lag"].size == 5
    assert r["gamma"].size == 5
    assert r["n_pairs"].sum() > 0
    ok = ~np.isnan(r["gamma"])
    assert np.all(r["gamma"][ok] >= 0)


def test_spsemv_rejects_bad_input():
    with pytest.raises(ValueError, match="same number of rows"):
        schabenberger_semivariogram_def(np.zeros((5, 2)), np.zeros(4))
