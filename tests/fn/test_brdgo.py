"""Tests for brdgo."""

import numpy as np
import pytest

from morie.fn.brdgo import bridge_observations


def test_brdgo_basic():
    rng = np.random.default_rng(1)
    ids = [f"L{i}" for i in range(6)]
    A = rng.normal(size=(6, 2))
    B = A * 3.0 + 1.0  # scale + shift, no rotation
    out = bridge_observations([dict(zip(ids, A)), dict(zip(ids, B))], ids)
    assert out["bridge_residual"] == pytest.approx(0.0, abs=1e-10)
    assert out["scale"] == pytest.approx(1 / 3, abs=1e-10)


def test_brdgo_edge():
    ids = ["a", "b", "c"]
    p = dict(zip(ids, np.eye(3, 2)))
    with pytest.raises(ValueError):
        bridge_observations([p, p], ["a", "z"])  # unknown bridge id
