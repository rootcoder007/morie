"""Tests for km083.kamath_ch6_ceat_random_effects."""

import math

from morie.fn import _array_core as np

from morie.fn.km083 import kamath_ch6_ceat_random_effects


def test_km083_basic():
    """Test basic functionality with several contexts."""
    rng = np.random.default_rng(42)
    N_contexts = 5
    # Each context: a small set of word vectors of dimension 3
    S_A1 = [rng.normal(0, 1, (4, 3)) for _ in range(N_contexts)]
    S_A2 = [rng.normal(0, 1, (4, 3)) for _ in range(N_contexts)]
    S_W1 = [rng.normal(0, 1, (4, 3)) for _ in range(N_contexts)]
    S_W2 = [rng.normal(0, 1, (4, 3)) for _ in range(N_contexts)]
    # Positive, finite weights (inverse-variance style)
    v = [abs(float(x)) for x in rng.normal(0, 1, N_contexts)]
    result = kamath_ch6_ceat_random_effects(S_A1, S_A2, S_W1, S_W2, v)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert "weat" in result
    assert len(result["weat"]) == N_contexts
    assert result["n"] == N_contexts


def test_km083_edge():
    """Test edge case: equal weights reduce to the plain mean of WEATs."""
    rng = np.random.default_rng(42)
    N_contexts = 3
    S_A1 = [rng.normal(0, 1, (4, 3)) for _ in range(N_contexts)]
    S_A2 = [rng.normal(0, 1, (4, 3)) for _ in range(N_contexts)]
    S_W1 = [rng.normal(0, 1, (4, 3)) for _ in range(N_contexts)]
    S_W2 = [rng.normal(0, 1, (4, 3)) for _ in range(N_contexts)]
    v = [1.0] * N_contexts  # equal weights
    result = kamath_ch6_ceat_random_effects(S_A1, S_A2, S_W1, S_W2, v)
    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    # With equal weights, the pooled estimate equals the plain mean
    expected = sum(result["weat"]) / N_contexts
    assert abs(result["estimate"] - expected) < 1e-9
