"""Tests for btnpb.boot_nonoverlap_block."""

from morie.fn import _array_core as np
from morie.fn.btnpb import boot_nonoverlap_block


def test_btnpb_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_nonoverlap_block(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_btnpb_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_nonoverlap_block(x)
    assert isinstance(result, dict)


def test_block_resampling_recomputed():
    """Non-overlapping blocks drawn with the documented generator."""
    import math

    import pytest

    x = np.array([0.5, 0.9, 0.4, 1.3, 1.1, 0.2, -0.3, 0.1, 0.8, 0.6, 1.5, 1.0])
    L, nb = 3, 4
    blocks = [[float(v) for v in x[i * L : (i + 1) * L]] for i in range(nb)]
    rng = np.random.default_rng(9)
    reps = []
    for _ in range(30):
        pick = rng.integers(0, nb, nb)
        vals = [v for k in pick for v in blocks[int(k)]]
        reps.append(sum(vals) / len(vals))
    r = boot_nonoverlap_block(x, block_len=L, B=30, seed=9)
    m = sum(reps) / 30
    assert r["se"] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in reps) / 29), rel=1e-12)
    assert r["n_blocks"] == 4
