"""Tests for sppow.schabenberger_power_variogram.

The book-certified identities for the whole variogram family live in
``test_schab_variogram_family.py``. This file pins the module's own
contract: it returns a real semivariogram, not the placeholder payload
it used to return.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.sppow import schabenberger_power_variogram


def test_sppow_returns_a_semivariogram():
    h = np.array([0.0, 1.0, 2.0, 4.0])
    r = schabenberger_power_variogram(h, nugget=0.0, c1=1.5, alpha=1.0)
    g = r["gamma"]
    assert g[0] == 0.0
    np.testing.assert_allclose(g[1:], [1.5, 3.0, 6.0], rtol=1e-12)
    assert r["model"] == "power"


def test_sppow_rejects_bad_input():
    with pytest.raises(ValueError, match="intrinsic hypothesis"):
        schabenberger_power_variogram(np.array([1.0]), 0.0, 1.0, 2.5)
