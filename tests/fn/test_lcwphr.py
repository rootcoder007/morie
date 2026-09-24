"""Tests for lcwphr.latent_class_weighted."""

from morie.fn import _array_core as np
import math
import pytest

from morie.fn.lcwphr import latent_class_weighted


def _make_binary_vector(rng, n):
    """Return a length-n binary vector (0.0/1.0) with both values present."""
    A = [float(v) for v in rng.integers(0, 2, n)]
    if not (any(v > 0.5 for v in A) and any(v < 0.5 for v in A)):
        # Fallback: alternate to guarantee both arms
        A = [1.0 if i % 2 == 0 else 0.0 for i in range(n)]
    return A


def _make_binary_matrix(rng, n, Q):
    """Return an (n, Q) binary matrix (0.0/1.0)."""
    raw = rng.integers(0, 2, (n, Q))
    H = [[float(v) for v in row] for row in raw]
    return H


def test_lcwphr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    Q = 3
    K = 2

    y = list(rng.normal(0.0, 1.0, n))
    A = _make_binary_vector(rng, n)
    H = _make_binary_matrix(rng, n, Q)

    result = latent_class_weighted(y, A, H, K)

    # RichResult is dict-like
    assert isinstance(result, dict)

    # Expected keys from docstring
    expected_keys = {
        "class_prevalence",
        "item_probabilities",
        "posterior",
        "class_ate",
        "ate",
        "naive_class_ate",
        "naive_ate",
    }
    assert expected_keys.issubset(result.keys())

    # Shape checks
    assert len(result["class_prevalence"]) == K
    assert len(result["item_probabilities"]) == K
    assert all(len(row) == Q for row in result["item_probabilities"])
    assert len(result["posterior"]) == n
    assert all(len(row) == K for row in result["posterior"])

    # class_ate / naive_class_ate are per-class (K values each)
    for key in ("class_ate", "naive_class_ate"):
        assert len(result[key]) == K
        for v in result[key]:
            assert isinstance(v, float)
            assert math.isfinite(v)

    # ate / naive_ate are scalars
    for key in ("ate", "naive_ate"):
        assert isinstance(result[key], float)
        assert math.isfinite(result[key])

    # Probabilities are finite and in [0, 1]
    for p in result["class_prevalence"]:
        assert math.isfinite(p)
        assert 0.0 <= p <= 1.0
    assert math.isclose(sum(result["class_prevalence"]), 1.0, abs_tol=1e-6)

    for row in result["item_probabilities"]:
        for p in row:
            assert math.isfinite(p)
            assert 0.0 <= p <= 1.0

    for row in result["posterior"]:
        for p in row:
            assert math.isfinite(p)
            assert 0.0 <= p <= 1.0


def test_lcwphr_edge():
    """Test that K < 1 raises a ValueError."""
    rng = np.random.default_rng(0)
    n = 40
    Q = 3

    y = list(rng.normal(0.0, 1.0, n))
    A = _make_binary_vector(rng, n)
    H = _make_binary_matrix(rng, n, Q)

    with pytest.raises(ValueError):
        latent_class_weighted(y, A, H, 0)
