"""Tests for depthH.halfspace_depth."""

from morie.fn import _array_core as np

from morie.fn.depthH import halfspace_depth


def test_depthH_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    theta = rng.normal(0, 1, 5)
    result = halfspace_depth(X, theta)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "count" in result
    assert "exact" in result
    assert "n" in result
    assert "p" in result

    # Compute the documented formula independently for verification.
    n, p = X.shape
    # reference direction set: the function uses directions to data points
    # and their negatives (Rousseeuw-Struef scheme for p > 2), which is an
    # upper bound.  We replicate the same scheme in the test.
    diffs = X - theta  # shape (n, p)
    # include both row and -row for each non-zero row
    dirs = []
    for i in range(n):
        row = diffs[i]
        if not (row == 0.0).all():
            dirs.append(row)
            dirs.append(-row)
    best = n
    for u in dirs:
        cnt = int((u @ diffs.T >= 0.0).sum())
        if cnt < best:
            best = cnt
    expected_estimate = best / n
    assert result["estimate"] == expected_estimate
    assert result["count"] == best
    assert result["n"] == n
    assert result["p"] == p
    # Above two dimensions the algorithm is not exact.
    assert result["exact"] == 0
    assert 0.0 <= result["estimate"] <= 1.0


def test_depthH_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    theta = rng.normal(0, 1, 5)
    result = halfspace_depth(X, theta)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "count" in result
    assert result["n"] == X.shape[0]
    assert result["p"] == X.shape[1]
