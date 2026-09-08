"""Tests for spkfun.schabenberger_k_function.

Book identities for the point-pattern family live in
test_schab_point_pattern.py. This pins the module's own contract.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spkfun import schabenberger_k_function

REGION = (0.0, 0.0, 10.0, 10.0)


def _pattern(seed=0, n=300):
    return np.random.default_rng(seed).random((n, 2)) * 10.0


def test_spkfun_returns_a_real_estimate():
    r = np.linspace(0.1, 1.0, 5)
    out = schabenberger_k_function(_pattern(), r=r, region=REGION)
    assert out["k"].shape == r.shape
    np.testing.assert_allclose(out["k_csr"], np.pi * r**2, rtol=1e-12)
    assert np.all(np.diff(out["k"]) > 0)          # K is increasing
    assert out["lambda_est"] == pytest.approx(3.0)


def test_spkfun_rejects_bad_input():
    with pytest.raises(ValueError, match="non-negative"):
        schabenberger_k_function(_pattern(), r=np.array([-1.0]), region=REGION)
