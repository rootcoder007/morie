"""Tests for spssoc.schabenberger_stationary_cov_semivario.

The book-certified identities for the whole variogram family live in
``test_schab_variogram_family.py``. This file pins the module's own
contract: it returns a real semivariogram, not the placeholder payload
it used to return.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spssoc import schabenberger_stationary_cov_semivario


def test_spssoc_returns_a_semivariogram():
    h = np.array([0.0, 1.0, 2.0])
    r = schabenberger_stationary_cov_semivario(
        lambda x: 2.0 * np.exp(-np.asarray(x, dtype=float)), h
    )
    assert r["sill"] == pytest.approx(2.0)
    assert r["gamma"][0] == pytest.approx(0.0)
    np.testing.assert_allclose(r["gamma"], r["sill"] - r["covariance"], rtol=1e-12)


def test_spssoc_rejects_bad_input():
    with pytest.raises(TypeError, match="callable"):
        schabenberger_stationary_cov_semivario("not a function", np.array([1.0]))
