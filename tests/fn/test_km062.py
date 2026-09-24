"""Tests for km062.kamath_ch4_krona_tuned_weights."""

from morie.fn import _array_core as np

from morie.fn.km062 import kamath_ch4_krona_tuned_weights


def test_km062_basic():
    """Test basic functionality using the docstring example."""
    W = [[1.0, 1.0], [1.0, 1.0]]
    A_k = [[1.0, 0.0], [0.0, 1.0]]
    B_k = [[1.0]]
    s = 2.0
    result = kamath_ch4_krona_tuned_weights(W, A_k, B_k, s)
    assert isinstance(result, dict)
    assert result["W_tuned"] == [[3.0, 1.0], [1.0, 3.0]]
    assert "delta" in result
    assert result["s"] == 2.0
    assert result["shape"] == (2, 2)
    assert result["estimate"] == 3.0
    assert result["n"] == 4
    assert result["method"] == "merged KronA weights (Kamath Eq 4.9)"


def test_km062_edge():
    """Test edge cases - s=0 returns W unchanged (identity)."""
    rng = np.random.default_rng(123)
    A_k = rng.normal(0, 1, (2, 2))
    B_k = rng.normal(0, 1, (1, 1))
    W = [[float(v) for v in row] for row in rng.normal(0, 1, (2, 2))]
    s = 0.0
    result = kamath_ch4_krona_tuned_weights(W, A_k, B_k, s)
    assert isinstance(result, dict)
    assert "W_tuned" in result
    assert result["s"] == 0.0
    assert result["shape"] == (2, 2)
    # When s=0, W_tuned should equal W (element-wise) - the merge is identity
    assert result["W_tuned"] == W
    # Delta should be all zeros
    for row in result["delta"]:
        for v in row:
            assert v == 0.0
