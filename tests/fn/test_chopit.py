"""Tests for chopit."""

import numpy as np
import pytest

from morie.fn.chopit import chopit_vignette


def test_chopit_basic():
    rng = np.random.default_rng(6)
    n = 300
    grp = np.repeat(["a", "b"], n // 2)
    shift = np.where(grp == "a", 0.0, 0.8)
    taus = np.array([-0.5, 0.5])
    mu_v = np.array([-0.8, 0.6])

    def rate(latent, sh):
        return 1 + (latent[:, None] > (taus[None, :] + sh[:, None])).sum(axis=1)

    Vg = np.column_stack([rate(mu_v[j] + rng.normal(size=n), shift) for j in range(2)])
    y = rate(rng.normal(size=n), shift)
    out = chopit_vignette(y, Vg, group=grp, n_categories=3)
    assert out["dif_shift"]["b"] == pytest.approx(0.8, abs=0.35)


def test_chopit_edge():
    with pytest.raises(ValueError):
        chopit_vignette([1, 2], np.ones((3, 2)))  # length mismatch
    with pytest.raises(ValueError):
        chopit_vignette([1, 5], np.ones((2, 2)), n_categories=3)  # rating out of range
