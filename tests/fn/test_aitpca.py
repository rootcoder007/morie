"""Tests for aitpca.aitchison_clr_pca."""

from morie.fn import _array_core as np

from morie.fn.aitpca import aitchison_clr_pca


def test_aitpca_basic():
    """Test basic functionality on a strictly positive compositional data set."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    # The function requires strictly positive compositions; shift so all entries > 0,
    # then normalise each row so the parts sum to 1 (a true composition).
    X = X - X.min(axis=1, keepdims=True) + 1.0
    X = X / X.sum(axis=1, keepdims=True)

    n, D = 100, 5
    k = 2

    result = aitchison_clr_pca(X, k)

    assert isinstance(result, dict)

    # The function returns a RichResult payload with these documented keys.
    for key in ("values", "loadings", "scores", "prop_var", "cum_prop", "k", "n", "D"):
        assert key in result, f"missing key: {key}"

    assert result["n"] == n
    assert result["D"] == D
    assert result["k"] == k

    # `values` is the full eigenspectrum of the clr covariance (length D).
    assert len(result["values"]) == D
    # `loadings` has shape (D, k): D components × k retained eigenvectors.
    assert len(result["loadings"]) == D
    assert all(len(row) == k for row in result["loadings"])
    # `scores` has shape (n, k).
    assert len(result["scores"]) == n
    assert all(len(row) == k for row in result["scores"])

    # Independent check of `prop_var` from the documented formula:
    # proportion retained = (s_1^2 + ... + s_r^2) / sum(s^2).
    vals = list(result["values"])
    pos = [v for v in vals if v > 0]
    tot = sum(v for v in pos)
    # With k=2 and D=5 we retain the two leading eigenvalues.
    expected_prop_r = sum(vals[j] for j in range(k)) / tot
    assert abs(result["prop_var"][0] - vals[0] / tot) < 1e-12
    assert abs(result["prop_var"][1] - vals[1] / tot) < 1e-12
    assert abs(sum(result["prop_var"][:k]) - expected_prop_r) < 1e-12

    # cum_prop is the cumulative sum of prop_var (independent recomputation).
    cum = 0.0
    for j in range(k):
        cum += result["prop_var"][j]
        assert abs(result["cum_prop"][j] - cum) < 1e-12


def test_aitpca_edge():
    """Test edge cases: still strictly positive, with a different k."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    X = X - X.min(axis=1, keepdims=True) + 1.0
    X = X / X.sum(axis=1, keepdims=True)

    k = 5
    result = aitchison_clr_pca(X, k)

    assert isinstance(result, dict)
    assert result["k"] == k
    assert result["n"] == 100
    assert result["D"] == 5
    assert len(result["loadings"]) == 5
    assert all(len(row) == k for row in result["loadings"])
    assert len(result["scores"]) == 100
