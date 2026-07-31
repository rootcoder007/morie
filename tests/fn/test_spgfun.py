"""Tests for spgfun.schabenberger_g_function.

Book identities for the point-pattern family live in
test_schab_point_pattern.py. This pins the module's own contract.
"""

import numpy as np
import pytest

from morie.fn.spgfun import schabenberger_g_function

REGION = (0.0, 0.0, 10.0, 10.0)


def _pattern(seed=0, n=300):
    return np.random.default_rng(seed).random((n, 2)) * 10.0


def test_spgfun_returns_a_real_estimate():
    out = schabenberger_g_function(_pattern(), region=REGION)
    assert np.all(np.diff(out["g"]) >= 0)
    assert out["g"][-1] == pytest.approx(1.0)
    assert out["nn_distances"].size == 300
    assert out["mean_nn"] > 0


def test_spgfun_rejects_bad_input():
    with pytest.raises(ValueError, match="at least two events"):
        schabenberger_g_function(np.array([[0.0, 0.0]]))
