"""Tests for medstg.sequential_mediation."""

from morie.fn import _array_core as np
import pytest

from morie.fn.medstg import sequential_mediation


def _chain(seed=42, n=3000):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m1 = 0.6 * x + rng.normal(scale=0.6, size=n)
    m2 = 0.4 * x + 0.5 * m1 + rng.normal(scale=0.6, size=n)
    y = 0.3 * x + 0.7 * m1 + 0.9 * m2 + rng.normal(scale=0.6, size=n)
    return x, m1, m2, y


def test_medstg_basic():
    out = sequential_mediation(*_chain())
    assert out["direct"] == pytest.approx(0.3, abs=0.06)
    assert out["serial"] == pytest.approx(0.6 * 0.5 * 0.9, abs=0.06)
    assert out["total"] == pytest.approx(out["direct"] + out["indirect_total"])


def test_medstg_edge():
    with pytest.raises(ValueError):
        sequential_mediation([1.0, 2.0], [1.0], [1.0, 2.0], [1.0, 2.0])  # length mismatch
