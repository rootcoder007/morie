"""Tests for spnsr.schabenberger_nugget_sill_range_effect.

Book identities for the kriging family live in test_schab_kriging.py.
This pins the module's own contract.
"""

import numpy as np
import pytest

from morie.fn.spnsr import schabenberger_nugget_sill_range_effect

CM = {"nugget": 0.0, "sill": 1.0, "range": 2.0, "model": "exponential"}


def _field(n=20):
    rng = np.random.default_rng(0)
    c = rng.random((n, 2)) * 5.0
    return c, np.sin(c[:, 0]) + np.cos(c[:, 1])


def test_spnsr_returns_a_real_estimate():
    r = schabenberger_nugget_sill_range_effect(0.1, 1.0, 1.0)
    assert np.isfinite(r["prediction"])
    assert r["variance"] >= 0
    assert r["weights"].size == 5
    assert r["weight_spread"] >= 0


def test_spnsr_rejects_bad_input():
    with pytest.raises(ValueError, match="`range` must be"):
        schabenberger_nugget_sill_range_effect(0.0, 1.0, -1.0)
