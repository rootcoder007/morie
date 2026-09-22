"""Tests for brcls.brier_score."""

from morie.fn import _array_core as np

from morie.fn.brcls import brier_score


def test_brcls_basic():
    """Test basic functionality against the documented formula."""
    rng_p = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)

    T, C = 5, 3
    # Predicted category probabilities: one row per observation, C columns.
    # Each row is normalised so it is a valid categorical distribution.
    raw = rng_p.uniform(0.0, 1.0, (T, C))
    row_sums = raw.sum(axis=1)
    P = raw / row_sums[:, None]

    # Observed category index, 1-based, length T, in [1, C].
    y = [int(v) for v in rng_y.integers(1, C + 1, T)]

    result = brier_score(P, y)

    # The function returns a RichResult (dict-like) with documented keys.
    assert isinstance(result, dict)
    assert "brier" in result
    assert "brier_scaled" in result
    assert "n" in result
    assert "C" in result
    assert result["n"] == T
    assert result["C"] == C

    # Independent recomputation from the docstring formula:
    # BS = (1/T) * sum_i sum_c ( p_ic - 1{y_i == c} )^2
    total = 0.0
    for i in range(T):
        for c in range(C):
            d = 1.0 if y[i] == c + 1 else 0.0
            total += (P[i][c] - d) ** 2
    expected_bs = total / T

    assert abs(result["brier"] - expected_bs) < 1e-12
    assert abs(result["brier_scaled"] - expected_bs / 2.0) < 1e-12


def test_brcls_edge():
    """Smoke test: valid minimal inputs still return the documented keys."""
    # Two observations, two classes, deterministic.
    P = [[0.7, 0.3],
         [0.2, 0.8]]
    y = [1, 2]

    result = brier_score(P, y)

    assert isinstance(result, dict)
    assert "brier" in result
    assert "brier_scaled" in result
    assert result["n"] == 2
    assert result["C"] == 2
