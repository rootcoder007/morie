"""Tests for spanis.schabenberger_geometric_anisotropy.

Book identities live in test_schab_matern_family.py.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spanis import schabenberger_geometric_anisotropy


def test_spanis_returns_corrected_and_raw_semivariograms():
    rng = np.random.default_rng(9)
    coords = rng.random((150, 2))
    z = 3.0 * coords[:, 0] + rng.normal(0, 0.1, 150)
    A = np.array([[1.0, 0.0], [0.0, 2.0]])
    r = schabenberger_geometric_anisotropy(coords, z, A, n_bins=6)
    assert r["lag"].size == 6
    assert r["gamma"].size == 6
    assert r["gamma_raw"].size == 6
    np.testing.assert_allclose(r["coords_corrected"], coords @ A.T, rtol=1e-12)
    ok = ~np.isnan(r["gamma"])
    assert np.all(r["gamma"][ok] >= 0)


def test_spanis_rejects_wrong_shaped_map():
    rng = np.random.default_rng(9)
    with pytest.raises(ValueError, match="must be"):
        schabenberger_geometric_anisotropy(
            rng.random((20, 2)), rng.normal(size=20), np.eye(3)
        )
