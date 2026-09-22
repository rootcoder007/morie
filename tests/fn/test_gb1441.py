"""Tests for gb1441.gibbons_fisher_exact."""

from morie.fn import _array_core as np

from morie.fn.gb1441 import gibbons_fisher_exact


def test_gb1441_basic():
    """Test basic functionality."""
    table = [[8, 2], [1, 5]]
    result = gibbons_fisher_exact(table)
    # The result is a RichResult, support dict-style access.
    assert isinstance(result, dict) or hasattr(result, "__getitem__")

    # Required keys per the docstring
    for key in ("p_value", "p_greater", "p_less", "prob", "statistic",
                "support", "method"):
        assert key in result, f"missing key: {key}"

    # Independent recomputation of the four core quantities.
    a, b = int(table[0][0]), int(table[0][1])
    c, d = int(table[1][0]), int(table[1][1])
    r1 = a + b
    r2 = c + d
    c1 = a + c
    N = r1 + r2

    from math import comb

    def hyper(k):
        return comb(r1, k) * comb(r2, c1 - k) / comb(N, c1)

    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    probs = {k: hyper(k) for k in range(lo, hi + 1)}
    pobs = probs[a]
    pg = sum(p for k, p in probs.items() if k >= a)
    pl = sum(p for k, p in probs.items() if k <= a)
    pv = sum(p for p in probs.values() if p <= pobs * (1.0 + 1e-12))
    pv = float(min(1.0, pv))

    assert result["statistic"] == a
    assert result["support"] == [lo, hi]
    assert abs(result["prob"] - pobs) < 1e-12
    assert abs(result["p_greater"] - pg) < 1e-12
    assert abs(result["p_less"] - pl) < 1e-12
    assert abs(result["p_value"] - pv) < 1e-12
    assert result["method"].startswith("Fisher exact")


def test_gb1441_edge():
    """Test edge cases (identical 2x2 input; deterministic outcome)."""
    table = [[8, 2], [1, 5]]
    result = gibbons_fisher_exact(table)
    # Re-uses a real 2x2 table rather than a 1-D array of floats.
    assert isinstance(result, dict) or hasattr(result, "__getitem__")
    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0
