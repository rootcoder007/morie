"""Tests for spffun.schabenberger_f_function.

Book identities for the point-pattern family live in
test_schab_point_pattern.py. This pins the module's own contract.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spffun import schabenberger_f_function

REGION = (0.0, 0.0, 10.0, 10.0)


def _pattern(seed=0, n=300):
    return np.random.default_rng(seed).random((n, 2)) * 10.0


def test_spffun_returns_a_real_estimate():
    out = schabenberger_f_function(_pattern(), REGION, n_grid=20)
    assert np.all(np.diff(out["f"]) >= 0)
    assert np.all((out["f"] >= 0) & (out["f"] <= 1))
    assert out["empty_space_distances"].size == 400


def test_spffun_rejects_bad_input():
    with pytest.raises(ValueError, match="positive area"):
        schabenberger_f_function(_pattern(), (0.0, 0.0, 0.0, 1.0))
