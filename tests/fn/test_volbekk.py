"""Tests for volbekk."""

from morie.fn import _array_core as np
import pytest

from morie.fn.volbekk import vol_bekk_garch


def _panel(seed=0, T=300, k=3):
    rng = np.random.default_rng(seed)
    cov = np.full((k, k), 0.5) + 0.5 * np.eye(k)
    return rng.multivariate_normal(np.zeros(k), cov, size=T)


def test_volbekk_basic():
    out = vol_bekk_garch(_panel())
    assert out["H"].shape == (300, 3, 3)
    assert np.all(np.linalg.eigvalsh(out["H"][100]) > 0)  # PD by construction
    assert 0 < out["persistence"] < 1


def test_volbekk_edge():
    with pytest.raises(ValueError):
        vol_bekk_garch(_panel()[:, :1])  # needs >= 2 series
    with pytest.raises(ValueError):
        vol_bekk_garch(_panel(T=10))  # too short
