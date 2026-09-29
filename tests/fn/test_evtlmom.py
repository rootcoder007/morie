"""Tests for evtlmom.evt_trimmed_lmom."""

from morie.fn import _array_core as np
from morie.fn.evtlmom import evt_trimmed_lmom


def test_evtlmom_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_trimmed_lmom(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_evtlmom_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_trimmed_lmom(x)
    assert isinstance(result, dict)


def test_untrimmed_lmoments_and_tl11_recomputed():
    """TL(0,0) gives Hosking's l1 = mean and l2 = 2 b1 - b0; TL(1,1) l1 is
    the Elamir-Seheult weighted mean sum (i-1)(n-i) x_(i) / (6 C(n,3))."""
    import math

    import pytest

    x = [3.1, -0.4, 2.2, 5.9, 1.7, 0.3, 4.4, 2.8]
    xs = sorted(x)
    n = 8
    b1 = sum((j - 1) / (n - 1) * v for j, v in enumerate(xs, start=1)) / n
    r = evt_trimmed_lmom(x, order=2)
    assert r["lambda"] == pytest.approx([sum(x) / n, 2 * b1 - sum(x) / n], rel=1e-12)
    t = evt_trimmed_lmom(x, s=1, t=1, order=1)
    want = sum((i - 1) * (n - i) * v for i, v in enumerate(xs, start=1)) / (math.comb(n, 3) * 1.0)
    assert t["lambda"][0] == pytest.approx(want, rel=1e-12)
