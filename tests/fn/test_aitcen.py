"""Tests for aitcen.aitchison_center."""

from morie.fn import _array_core as np

from morie.fn.aitcen import aitchison_center


def test_aitcen_basic():
    """Test basic functionality with strictly positive (compositional) data."""
    rng = np.random.default_rng(42)
    X = np.exp(rng.normal(0.0, 1.0, (100, 5)))
    result = aitchison_center(X)
    assert isinstance(result, dict)
    assert "center" in result
    assert "geometric_mean" in result
    assert "n" in result and "D" in result

    # Independent computation of the closed geometric mean from the documented
    # formula: xi-hat = C(g_1, ..., g_D), g_i = exp(mean(log X[:, i]))
    n, D = X.shape
    g_expected = [
        float(np.exp(np.sum(np.log(X[:, j])) / n))
        for j in range(D)
    ]
    s = float(np.sum(g_expected))
    center_expected = [v / s for v in g_expected]  # total=1.0

    assert result["n"] == n
    assert result["D"] == D

    gm = result["geometric_mean"]
    assert len(gm) == D
    for ge, gg in zip(g_expected, gm):
        assert abs(ge - float(gg)) < 1e-12

    center = result["center"]
    assert len(center) == D
    for ce, cg in zip(center_expected, center):
        assert abs(ce - float(cg)) < 1e-12
    # Closure sums to total=1.0
    assert abs(float(np.sum(center)) - 1.0) < 1e-12


def test_aitcen_edge():
    """Test edge cases with strictly positive data and custom total."""
    rng = np.random.default_rng(42)
    X = np.exp(rng.normal(0.0, 1.0, (100, 5)))
    result = aitchison_center(X, total=100.0)
    assert isinstance(result, dict)
    assert "center" in result
    assert "geometric_mean" in result
    # With total=100, the closed centre should sum to 100
    assert abs(float(np.sum(result["center"])) - 100.0) < 1e-12
