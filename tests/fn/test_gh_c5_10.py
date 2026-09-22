"""Tests for gh_c5_10.ghosal_poi_ker."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c5_10 import ghosal_poi_ker


def test_gh_c5_10_basic():
    """Test basic functionality with default stick-breaking draw."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_poi_ker(x)
    assert "estimate" in result
    assert "pmf" in result
    assert "method" in result
    est = float(result["estimate"])
    assert math.isfinite(est)
    assert 0.0 <= est <= 1.0
    pmf = [float(v) for v in result["pmf"]]
    assert len(pmf) == 5
    for v in pmf:
        assert math.isfinite(v)
        assert 0.0 <= v <= 1.0


def test_gh_c5_10_edge():
    """Test edge cases with supplied atoms/weights against the documented formula."""
    ks = [0.0, 3.0, 7.0]
    lambdas = [0.5, 2.0, 4.0]
    weights = [0.2, 0.3, 0.5]
    result = ghosal_poi_ker(ks, lambdas=lambdas, weights=weights,
                            alpha=1.0, n_terms=100, seed=42)

    pmf = [float(v) for v in result["pmf"]]
    assert len(pmf) == len(ks)

    # Independent recomputation of f(k) = sum_j w_j * Poi(k; lambda_j)
    expected = []
    for k, lvec in zip(ks, [lambdas] * len(ks)):
        k_int = int(k)
        s = 0.0
        for w, lam in zip(weights, lambdas):
            s += w * math.exp(-lam + k_int * math.log(lam)
                              - math.lgamma(k_int + 1.0))
        expected.append(s)

    # `estimate` is documented to be the mixture evaluated at the first k.
    est = float(result["estimate"])
    assert math.isclose(est, expected[0], rel_tol=1e-12, abs_tol=1e-12)
    for got, exp in zip(pmf, expected):
        assert math.isclose(got, exp, rel_tol=1e-12, abs_tol=1e-12)
