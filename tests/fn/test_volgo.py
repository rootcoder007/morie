"""Tests for volgo."""

from morie.fn import _array_core as np
import pytest

from morie.fn.volgo import vol_garch_orthogonal


def _panel(seed=0, T=300, k=3):
    rng = np.random.default_rng(seed)
    cov = np.full((k, k), 0.5) + 0.5 * np.eye(k)
    return rng.multivariate_normal(np.zeros(k), cov, size=T)


def test_volgo_basic():
    out = vol_garch_orthogonal(_panel())
    assert out["full_rank"] is True
    assert np.allclose(out["H"][50], out["H"][50].T)


def test_volgo_edge():
    # truncating to k < d makes H singular, and the flag says so
    red = vol_garch_orthogonal(_panel(), k=1)
    assert red["full_rank"] is False
    assert np.linalg.matrix_rank(red["H"][50]) == 1
    with pytest.raises(ValueError):
        vol_garch_orthogonal(_panel(), k=9)
