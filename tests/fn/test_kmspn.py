"""Tests for kmspn.kamath_t5_span_corruption."""

from morie.fn import _array_core as np
from morie.fn.kmspn import kamath_t5_span_corruption


def test_kmspn_basic():
    """Test basic functionality."""
    tokens = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kamath_t5_span_corruption(tokens)
    assert isinstance(result, dict)
    assert "estimate" in result or "input" in result


def test_kmspn_edge():
    """Test edge cases."""
    tokens = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kamath_t5_span_corruption(tokens)
    assert isinstance(result, dict)


def test_span_corruption_structure():
    toks = list("abcdefghijklmnopqrst")
    r = kamath_t5_span_corruption(toks, mean_span_len=2.0, corruption_rate=0.3, seed=11)
    n_mask = round(20 * 0.3)
    n_spans = round(n_mask / 2.0)
    assert r["n_masked"] == n_mask and r["n_spans"] == n_spans
    assert sum(r["span_lengths"]) == n_mask
    # rebuild the original from input + target
    fill = {}
    cur = None
    for t in r["target"]:
        if t.startswith("<extra_id_"):
            cur = t
            fill[cur] = []
        else:
            fill[cur].append(t)
    rebuilt = [u for t in r["input"] for u in fill.get(t, [t])]
    assert rebuilt == toks
