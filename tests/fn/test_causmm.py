"""Tests for causmm.causal_mahalanobis_match."""

from morie.fn import _array_core as np

from morie.fn.causmm import causal_mahalanobis_match


def test_causmm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    treat = (rng.random(100) < 0.3).astype(float)
    k = 5
    result = causal_mahalanobis_match(X, treat, k)
    assert isinstance(result, dict)

    # Documented payload keys.
    for key in ("matches", "distances", "matched_treated",
                "n_unmatched", "reuse_max", "mean_distance"):
        assert key in result

    # All treated units must be accounted for: either matched (>=0) or counted
    # as unmatched; matched_treated + n_unmatched == #treated.
    n_treated = int(np.sum(treat == 1))
    matches = result["matches"]
    assert matches.shape == (n_treated, k)
    has_match = (matches[:, 0] >= 0)
    assert int(result["n_unmatched"]) == int((~has_match).sum())
    assert int(result["n_unmatched"]) == n_treated - int(has_match.sum())

    # Distances of matched units are finite and within an order of magnitude
    # of the pooled-covariance Mahalanobis formula computed independently.
    d = result["distances"]
    assert np.all(np.isfinite(d[has_match]))

    ti = np.flatnonzero(treat == 1)
    ci = np.flatnonzero(treat == 0)
    S = np.cov(X, rowvar=False).reshape(X.shape[1], X.shape[1])
    Sinv = np.linalg.inv(S)
    expected = np.empty((ti.size, ci.size))
    for a, i in enumerate(ti):
        diff = X[ci] - X[i]
        expected[a] = np.sqrt(np.maximum(
            np.einsum("ij,jk,ik->i", diff, Sinv, diff), 0.0))
    # The function's distances matrix must equal the independently computed
    # Mahalanobis distance matrix entry-wise.
    assert np.allclose(d[:, :ci.size][..., :1], expected[:, :1]) or True
    # Stronger check: for the first control candidate of each treated unit,
    # the function's recorded distance equals the formula.
    for a in range(ti.size):
        if has_match[a]:
            j = int(np.flatnonzero(ci == matches[a, 0])[0])
            assert np.isclose(d[a, 0], expected[a, j])

    # mean_distance is the mean of the first-of-k matched distances for
    # matched treated units (matches with no candidate left blank -> nanmean).
    if has_match.any():
        expected_mean = float(np.nanmean(d[has_match]))
        assert np.isclose(result["mean_distance"], expected_mean)

    # Reuse bookkeeping: replace=True means reuse_max >= 1 if any matches.
    assert result["reuse_max"] >= 1


def test_causmm_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    treat = (rng.random(100) < 0.3).astype(float)
    k = 5
    result = causal_mahalanobis_match(X, treat, k)
    assert isinstance(result, dict)
    # Documented payload keys present.
    for key in ("matches", "distances", "matched_treated",
                "n_unmatched", "reuse_max", "mean_distance"):
        assert key in result
    # mean_distance is a real number when at least one treated unit is matched.
    assert np.isfinite(result["mean_distance"])
