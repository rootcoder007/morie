"""Tests for zxcpv."""

import numpy as np
import pytest

from morie.fn.zxcpv import copula_vine_sp

def test_zxcpv_basic():
    rng = np.random.default_rng(42)
    z = rng.normal(size=300)
    X = np.column_stack([0.9 * z + 0.4 * rng.normal(size=300), z,
                         0.9 * z + 0.4 * rng.normal(size=300)])
    out = copula_vine_sp(X)
    assert out["root"] == 1  # the hub variable
    assert len(out["tree1_edges"]) == 2


def test_zxcpv_edge():
    with pytest.raises(ValueError):
        copula_vine_sp(np.zeros((10, 1)))
