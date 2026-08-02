"""Tests for spnest.schabenberger_nested_variogram.

The book-certified identities for the whole variogram family live in
``test_schab_variogram_family.py``. This file pins the module's own
contract: it returns a real semivariogram, not the placeholder payload
it used to return.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spnest import schabenberger_nested_variogram


def test_spnest_returns_a_semivariogram():
    h = np.array([0.0, 0.5, 2.0])
    r = schabenberger_nested_variogram(
        h,
        [{"model": "nugget", "sill": 0.2},
         {"model": "spherical", "sill": 1.0, "range": 1.5}],
    )
    assert r["total_sill"] == pytest.approx(1.2)
    assert len(r["components"]) == 2
    assert r["gamma"][0] == 0.0
    assert r["gamma"][-1] == pytest.approx(1.2)


def test_spnest_rejects_bad_input():
    with pytest.raises(ValueError, match="at least one component"):
        schabenberger_nested_variogram(np.array([1.0]), [])
