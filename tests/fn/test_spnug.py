"""Tests for spnug.schabenberger_nugget_effect.

The book-certified identities for the whole variogram family live in
``test_schab_variogram_family.py``. This file pins the module's own
contract: it returns a real semivariogram, not the placeholder payload
it used to return.
"""

import numpy as np
import pytest

from morie.fn.spnug import schabenberger_nugget_effect


def test_spnug_returns_a_semivariogram():
    h = np.array([0.0, 0.5, 2.0])
    r = schabenberger_nugget_effect(h, nugget=0.3, sill=1.0, range=1.0)
    assert r["gamma_at_zero"] == 0.0
    assert r["limit_at_zero_plus"] == pytest.approx(0.3)
    assert r["total_sill"] == pytest.approx(1.3)
    assert r["gamma"][0] == 0.0


def test_spnug_rejects_bad_input():
    with pytest.raises(ValueError):
        schabenberger_nugget_effect(np.array([1.0]), -0.5, 1.0, 1.0)
