"""Tests for crsfst.crs_forest."""

from morie.fn import _array_core as np

from morie.fn.crsfst import crs_forest


def test_crsfst_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    time = np.linspace(0, 10, n)
    # Event must be a binary indicator (0/1), not a continuous normal.
    event = rng.uniform(0, 1, n) < 0.7
    # Treatment D must be 0 or 1, with BOTH arms present in the sample.
    D = (rng.uniform(0, 1, n) < 0.5).astype(float)
    X = rng.normal(0, 1, (n, 5))
    # K is the number of cross-fitting folds (an int), not a 10x10 matrix.
    result = crs_forest(time, event, D, X, K=3)
    assert isinstance(result, dict)
    # The function returns a RichResult whose payload exposes these keys.
    assert "estimate" in result
    assert "cate" in result
    assert "rmst_treated" in result
    assert "rmst_control" in result
    assert "fold" in result
    assert "tau" in result
    assert "n_leaked" in result
    # The honest cross-fit must not leak training observations into the
    # test-fold predictions.
    assert result["n_leaked"] == 0
    # At least some fold must have produced a comparable pair of arms.
    assert result["n_scored"] > 0
    # Estimate and SE are scalars returned in the payload.
    assert np.isfinite(result["estimate"])
    # tau defaults to the smaller of the per-arm maximum observed times.
    t1 = float(max(time[i] for i in range(n) if D[i] == 1))
    t0 = float(max(time[i] for i in range(n) if D[i] == 0))
    assert result["tau"] == min(t1, t0)


def test_crsfst_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 100
    time = np.linspace(0, 10, n)
    event = rng.uniform(0, 1, n) < 0.7
    D = (rng.uniform(0, 1, n) < 0.5).astype(float)
    X = rng.normal(0, 1, (n, 5))
    result = crs_forest(time, event, D, X, K=3)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result
    assert "n" in result
    assert result["n"] == n
