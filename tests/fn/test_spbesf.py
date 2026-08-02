"""Tests for spbesf.schabenberger_bessel_function.

Book identities live in test_schab_matern_family.py.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spbesf import schabenberger_bessel_function


def test_spbesf_returns_k_nu():
    t = np.array([0.1, 1.0, 5.0])
    r = schabenberger_bessel_function(t, nu=1.5)
    v = r["value"]
    assert v.shape == t.shape
    assert np.all(v > 0)
    assert np.all(np.diff(v) < 0)          # K_nu decays monotonically
    assert r["nu"] == 1.5


def test_spbesf_rejects_non_positive_argument():
    with pytest.raises(ValueError, match="positive"):
        schabenberger_bessel_function(np.array([0.0]), 1.0)
