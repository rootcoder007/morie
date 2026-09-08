"""Tests for gb251 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb251 import gibbons_pit


def test_gb251_basic():
    # Rate over seeds, not a single p-value: under the true generator
    # the K-S p is Uniform(0,1), so any fixed seed drops below 0.01
    # about 1% of the time by construction (seed 1 gives 0.0066).
    rng = np.random.default_rng(1)
    ps = [gibbons_pit(rng.standard_normal(300))["ks_p"] for _ in range(20)]
    assert sum(p > 0.01 for p in ps) >= 18  # measured 20/20 at this seed


def test_gb251_edge():
    with pytest.raises(ValueError):
        gibbons_pit([np.inf])
