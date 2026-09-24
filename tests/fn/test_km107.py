"""Tests for km107.kamath_ch6_pii_likelihood."""

import math

from morie.fn import _array_core as np

from morie.fn.km107 import kamath_ch6_pii_likelihood


def test_km107_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    L_r = 3
    L_q = 2
    a_m = rng.uniform(0.01, 1.0, L_r)
    A = ["name", "phone"]
    x = ["contact", "John"]
    result = kamath_ch6_pii_likelihood(a_m, A, x, L_q, L_r)
    assert isinstance(result, dict)
    expected_keys = {"estimate", "log_likelihood", "per_token",
                     "context_lengths", "n_other_pii", "n", "method"}
    assert expected_keys <= set(result.keys())
    assert result["n"] == L_r
    assert result["n_other_pii"] == len(A)
    assert len(result["per_token"]) == L_r
    assert len(result["context_lengths"]) == L_r
    assert result["context_lengths"] == [L_q + r - 1 for r in range(1, L_r + 1)]
    assert math.isfinite(result["estimate"])
    assert 0 < result["estimate"] <= 1
    assert math.isfinite(result["log_likelihood"])


def test_km107_edge():
    """Test edge cases."""
    result = kamath_ch6_pii_likelihood([0.5, 0.25], ["name"],
                                       ["contact", "John"], 2, 2)
    assert isinstance(result, dict)
    expected_keys = {"estimate", "log_likelihood", "per_token",
                     "context_lengths", "n_other_pii", "n", "method"}
    assert expected_keys <= set(result.keys())
    assert result["estimate"] == 0.125
    assert result["context_lengths"] == [2, 3]
    assert result["n"] == 2
    assert result["n_other_pii"] == 1
    assert result["per_token"] == [0.5, 0.25]
    assert math.isfinite(result["log_likelihood"])
