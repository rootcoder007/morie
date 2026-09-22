"""Tests for alfdst.alphafold_distogram."""

from morie.fn import _array_core as np

from morie.fn.alfdst import alphafold_distogram


def test_alfdst_basic():
    """Test basic functionality."""
    n = 4
    cz = 3
    nb = 64
    rng = np.random.default_rng(44)
    # z: n x n x cz
    z = [[[rng.normal(0, 1) for _ in range(cz)] for _ in range(n)] for _ in range(n)]
    # w: nb x cz
    w = [[rng.normal(0, 1) for _ in range(cz)] for _ in range(nb)]
    result = alphafold_distogram(z, w)
    assert isinstance(result, dict)
    assert "p" in result
    assert "dist" in result
    assert "loss" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result

    # Output is symmetric in i and j
    for i in range(n):
        for j in range(n):
            assert result["p"][i][j] == result["p"][j][i]
            assert result["dist"][i][j] == result["dist"][j][i]

    # Each distribution sums to one
    for i in range(n):
        for j in range(n):
            assert abs(sum(result["p"][i][j]) - 1.0) < 1e-6

    # n is reported correctly
    assert result["n"] == n

    # No ground-truth distances -> loss is None
    assert result["loss"] is None

    # Estimate equals mean of dist
    flat = [result["dist"][i][j] for i in range(n) for j in range(n)]
    expected_estimate = sum(flat) / len(flat)
    assert abs(result["estimate"] - expected_estimate) < 1e-12


def test_alfdst_edge():
    """Test edge cases."""
    n = 3
    cz = 2
    nb = 64
    rng = np.random.default_rng(7)
    z = [[[rng.normal(0, 1) for _ in range(cz)] for _ in range(n)] for _ in range(n)]
    w = [[rng.normal(0, 1) for _ in range(cz)] for _ in range(nb)]
    result = alphafold_distogram(z, w)
    assert isinstance(result, dict)
    assert result["n"] == n
    # symmetry still holds at small n
    for i in range(n):
        for j in range(n):
            assert result["dist"][i][j] == result["dist"][j][i]
            assert abs(sum(result["p"][i][j]) - 1.0) < 1e-6
