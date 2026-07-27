"""Tests for vecmod."""

import numpy as np
import pytest

from morie.fn.vecmod import vector_error_correction

def _system(seed=0, n=400):
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.standard_normal(n))
    return np.column_stack([trend + rng.standard_normal(n) * 0.5,
                            2 * trend + rng.standard_normal(n) * 0.5])


def test_vecmod_basic():
    out = vector_error_correction(_system(), r=1)
    assert out["alpha"].shape == (2, 1)
    assert out["johansen_rank_5pct"] == 1
    assert np.max(np.abs(out["alpha"])) > 0.01  # something adjusts


def test_vecmod_edge():
    with pytest.raises(ValueError):
        vector_error_correction(_system(), r=5)
    with pytest.raises(ValueError):
        vector_error_correction(_system()[:20])  # too short
