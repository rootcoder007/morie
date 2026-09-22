"""Tests for divLst.diversity."""

from morie.fn import _array_core as np

from morie.fn.divLst import diversity


def _flat(M):
    """Row-major iteration helper for the numpy-like shim."""
    return [M[r][c] for r in range(len(M)) for c in range(len(M[r]))]


def test_divLst_basic():
    """Test basic functionality with known items and a symmetric sim matrix."""
    # Build a 5x5 symmetric similarity matrix with known entries.
    S = [
        [1.0, 0.2, 0.8, 0.5, 0.1],
        [0.2, 1.0, 0.3, 0.4, 0.6],
        [0.8, 0.3, 1.0, 0.7, 0.2],
        [0.5, 0.4, 0.7, 1.0, 0.9],
        [0.1, 0.6, 0.2, 0.9, 1.0],
    ]
    # Valid item indices (all within 0..4) referencing the 5x5 matrix.
    items = [0, 1, 2, 3, 4]

    result = diversity(items, S)

    # Keys per the docstring.
    assert "estimate" in result
    assert "ils" in result
    assert "n_pairs" in result
    assert "min_pair_sim" in result
    assert "max_pair_sim" in result

    # Enumerate the 10 pairs and compute everything independently.
    kk = len(items)
    pairs = [(a, b) for a in range(kk) for b in range(a + 1, kk)]
    s_vals = [S[items[a]][items[b]] for a, b in pairs]
    expected_n_pairs = len(pairs)
    expected_ils = sum(s_vals)
    expected_estimate = sum(1.0 - s for s in s_vals) / expected_n_pairs
    expected_min = min(s_vals)
    expected_max = max(s_vals)

    assert result["n_pairs"] == expected_n_pairs
    assert abs(result["ils"] - expected_ils) < 1e-9
    assert abs(result["estimate"] - expected_estimate) < 1e-9
    assert abs(result["min_pair_sim"] - expected_min) < 1e-9
    assert abs(result["max_pair_sim"] - expected_max) < 1e-9


def test_divLst_edge():
    """Test edge case: fewer than 2 items yields NaN estimate and zero pairs."""
    S = [[1.0, 0.5], [0.5, 1.0]]
    items = [0]

    result = diversity(items, S)

    assert "estimate" in result
    assert "n_pairs" in result
    assert result["n_pairs"] == 0
    assert result["estimate"] != result["estimate"]  # NaN check
