"""Tests for ksr060 (Kosorok shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr060 import kosorok_ch2_u_process_measure


def test_ksr060_basic():
    from itertools import combinations
    rng = np.random.default_rng(17)
    X = rng.random(40)
    out = kosorok_ch2_u_process_measure(lambda a, b: abs(a - b) / 2, X, m=2)
    assert out["U"] == pytest.approx(
        np.mean([abs(a - b) / 2 for a, b in combinations(X, 2)]))


def test_ksr060_edge():
    rng = np.random.default_rng(17)
    with pytest.raises(ValueError):
        kosorok_ch2_u_process_measure(lambda a, b, c: a, rng.random(400), m=3)
