"""Tests for medFront.front_door."""

from morie.fn import _array_core as np
import pytest

from morie.fn.fdadj import frontdoor_adjustment
from morie.fn.medFront import front_door


def test_medFront_basic():
    rng = np.random.default_rng(42)
    n = 6000
    u = (rng.random(n) < 0.5).astype(int)
    x = (rng.random(n) < 0.2 + 0.6 * u).astype(int)
    m = (rng.random(n) < 0.1 + 0.8 * x).astype(int)
    y = (rng.random(n) < 0.1 + 0.5 * m + 0.3 * u).astype(int)
    a = front_door(y, x, m)
    b = frontdoor_adjustment(x, m, y)
    assert a["distribution"] == b["distribution"]  # same estimator, y-first args


def test_medFront_edge():
    with pytest.raises(ValueError):
        front_door(np.zeros(10), np.zeros(5), np.zeros(10))  # length mismatch
