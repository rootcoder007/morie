"""Tests for ca12e1.ca_chapter_12_equation_1."""

from morie.fn import _array_core as np

from morie.fn.ca12e1 import ca_chapter_12_equation_1


def _build_row_standardized_weights(n, k=4, seed=0):
    """Build a row-standardized k-nearest-neighbor-style weight matrix."""
    rng = np.random.default_rng(seed)
    # Build a deterministic symmetric adjacency: connect i to its k nearest
    # neighbors along the index axis (wrap-around). This keeps the test
    # self-contained and reproducible.
    nbrs = np.zeros((n, n), dtype=float)
    for i in range(n):
        for d in range(1, k + 1):
            j = (i + d) % n
            nbrs[i, j] = 1.0
            nbrs[j, i] = 1.0
    row_sums = nbrs.sum(axis=1, keepdims=True)
    w = nbrs / row_sums
    return w


def test_ca12e1_basic():
    """Test basic functionality against the documented formula."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    n = x.shape[0]
    w = _build_row_standardized_weights(n, k=4, seed=1)

    result = ca_chapter_12_equation_1(x, w)

    # Headline key per the docstring is "value" (not "estimate"/"statistic").
    assert isinstance(result, dict)
    assert "value" in result

    # Independent recomputation of Moran's I from the documented formula:
    #   I = n * sum_ij w_ij (x_i - xbar)(x_j - xbar)
    #       / (W * sum_i (x_i - xbar)^2)
    # with W = sum_ij w_ij. For a row-standardized W, sum_ij w_ij = n.
    xbar = x.mean()
    dx = x - xbar
    s2 = float((dx ** 2).sum())
    # numerator: n * x' (W @ x), but subtract xbar terms carefully via dx.
    # n * sum_ij w_ij dx_i dx_j  ==  n * dx . (W @ dx)
    Wdx = w @ dx
    num = n * float((dx * Wdx).sum())
    Wsum = float(w.sum())
    expected = num / (Wsum * s2)

    got = float(result["value"])
    assert abs(got - expected) < 1e-8 * max(1.0, abs(expected))


def test_ca12e1_edge():
    """Test edge cases: shape and return-type contract still hold."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    n = x.shape[0]
    w = _build_row_standardized_weights(n, k=4, seed=1)

    result = ca_chapter_12_equation_1(x, w)
    assert isinstance(result, dict)
    assert "value" in result
    # The returned value must be a real number.
    val = result["value"]
    assert isinstance(val, float) or (hasattr(val, "item") and not hasattr(val, "__len__"))
