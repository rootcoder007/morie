"""Tests for spkwt.schabenberger_kriging_weights.

Book identities for the kriging family live in test_schab_kriging.py.
This pins the module's own contract.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spkwt import schabenberger_kriging_weights

CM = {"nugget": 0.0, "sill": 1.0, "range": 2.0, "model": "exponential"}


def _field(n=20):
    rng = np.random.default_rng(0)
    c = rng.random((n, 2)) * 5.0
    return c, np.sin(c[:, 0]) + np.cos(c[:, 1])


def test_spkwt_returns_a_real_estimate():
    Sigma = np.eye(4) * 3.0
    sig = np.array([1.0, 0.8, 0.4, 0.1])
    r = schabenberger_kriging_weights(Sigma, sig)
    np.testing.assert_allclose(Sigma @ r["weights"], sig, atol=1e-10)
    assert r["unbiased"] is False
    u = schabenberger_kriging_weights(Sigma, sig, unbiased=True)
    assert u["weight_sum"] == pytest.approx(1.0, abs=1e-12)


def test_spkwt_rejects_bad_input():
    with pytest.raises(ValueError, match="square"):
        schabenberger_kriging_weights(np.ones((2, 3)), np.ones(2))
