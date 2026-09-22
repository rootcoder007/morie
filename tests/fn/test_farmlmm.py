"""Tests for farmlmm.farm_cpu."""

from morie.fn import _array_core as np

from morie.fn.farmlmm import farm_cpu


def _to_python_2d(arr):
    """Convert a 2-D array-like to a list of lists of Python floats."""
    return [[float(v) for v in row] for row in arr]


def _to_python_1d(arr):
    """Convert a 1-D array-like to a list of Python floats."""
    return [float(v) for v in arr]


def test_farmlmm_basic():
    """Test basic functionality against the documented formula."""
    rng = np.random.default_rng(43)

    n = 100
    p = 5

    # y: response vector of length n
    y = _to_python_1d(rng.normal(0, 1, n))

    # G: genotype matrix of shape (n, p)
    G = _to_python_2d(rng.normal(0, 1, (n, p)))

    # Per the docstring: threshold defaults to 0.05 / p.
    expected_threshold = 0.05 / p

    result = farm_cpu(y, G)

    assert isinstance(result, dict)

    # Documented return keys.
    assert "estimate" in result
    assert "selected" in result
    assert "p" in result
    assert "iterations" in result
    assert "converged" in result
    assert "oscillating" in result
    assert "threshold" in result
    assert "history" in result
    assert "method" in result

    # Threshold must equal 0.05 / p, computed independently in the test.
    assert result["threshold"] == expected_threshold

    # estimate and selected must always agree (the model returns the same list).
    assert result["estimate"] == result["selected"]

    # p must be a per-marker p-value vector of length p.
    assert len(result["p"]) == p

    # iterations is the number of FEM rounds actually executed (>= 1).
    assert int(result["iterations"]) >= 1

    # History length must equal the iteration count.
    assert len(result["history"]) == int(result["iterations"])

    # Method string is fixed by the implementation.
    assert isinstance(result["method"], str)
    assert "FarmCPU" in result["method"]

    # Selected entries, if any, must be valid marker indices.
    for j in result["selected"]:
        assert 0 <= int(j) < p


def test_farmlmm_edge():
    """Edge case: small design with seed-driven randomness."""
    rng = np.random.default_rng(7)

    n = 20
    p = 3

    y = _to_python_1d(rng.normal(0, 1, n))
    G = _to_python_2d(rng.normal(0, 1, (n, p)))

    result = farm_cpu(y, G, max_iter=3, seed=1)

    assert isinstance(result, dict)
    assert result["estimate"] == result["selected"]
    assert len(result["p"]) == p
    assert int(result["iterations"]) >= 1
    assert len(result["history"]) == int(result["iterations"])
