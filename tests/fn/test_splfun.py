"""Tests for splfun.schabenberger_l_function.

Book identities for the point-pattern family live in
test_schab_point_pattern.py. This pins the module's own contract.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.splfun import schabenberger_l_function

REGION = (0.0, 0.0, 10.0, 10.0)


def _pattern(seed=0, n=300):
    return np.random.default_rng(seed).random((n, 2)) * 10.0


def test_splfun_returns_a_real_estimate():
    r = np.linspace(0.1, 1.0, 5)
    out = schabenberger_l_function(_pattern(), r=r, region=REGION)
    np.testing.assert_allclose(out["l"], np.sqrt(np.maximum(out["k"], 0) / np.pi),
                               rtol=1e-12)
    np.testing.assert_allclose(out["l_minus_r"], out["l"] - r, rtol=1e-12)


def test_splfun_rejects_bad_input():
    with pytest.raises(ValueError, match="positive area"):
        schabenberger_l_function(_pattern(), region=(0.0, 0.0, 1.0, 0.0))
