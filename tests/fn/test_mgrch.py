"""Tests for mgrch."""

import numpy as np
import pytest

from morie.fn.mgrch import bekk_garch_multivariate


def _panel(seed=0, T=300, k=3):
    rng = np.random.default_rng(seed)
    cov = np.full((k, k), 0.5) + 0.5 * np.eye(k)
    return rng.multivariate_normal(np.zeros(k), cov, size=T)


def test_mgrch_basic():
    out = bekk_garch_multivariate(_panel())
    assert out["H"].shape == (300, 3, 3)
    assert np.all(np.linalg.eigvalsh(out["H"][100]) > 0)  # PD by construction
    assert 0 < out["persistence"] < 1


def test_mgrch_edge():
    with pytest.raises(ValueError):
        bekk_garch_multivariate(_panel()[:, :1])  # needs >= 2 series
    with pytest.raises(ValueError):
        bekk_garch_multivariate(_panel(T=10))  # too short
