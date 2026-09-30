import math

import pytest

from morie.fn.research_ecological import ecological_decompose


def _pcov(a, b):
    n = len(a)
    return math.fsum(u * v for u, v in zip(a, b)) / n - (math.fsum(a) / n) * (math.fsum(b) / n)


X = [math.sin(1.7 * i) * 3 + i % 4 for i in range(37)]
Y = [math.cos(0.9 * i) + 0.3 * X[i] - (i % 3) for i in range(37)]
G = [f"g{i % 5}" for i in range(37)]


def test_decomposition_is_exact_and_matches_direct_formulas():
    d = ecological_decompose(X, Y, G)
    means = {}
    for g in set(G):
        ix = [i for i in range(37) if G[i] == g]
        means[g] = (math.fsum(X[i] for i in ix) / len(ix), math.fsum(Y[i] for i in ix) / len(ix))
    bx = [means[g][0] for g in G]
    by = [means[g][1] for g in G]
    assert d.cov["individual"] == pytest.approx(_pcov(X, Y), abs=1e-12)
    assert d.cov["between"] == pytest.approx(_pcov(bx, by), abs=1e-12)
    assert d.cov["individual"] == pytest.approx(d.cov["between"] + d.cov["within"], abs=1e-12)
    assert d.var_x["individual"] == pytest.approx(d.var_x["between"] + d.var_x["within"], abs=1e-12)
    assert d.var_y["individual"] == pytest.approx(d.var_y["between"] + d.var_y["within"], abs=1e-12)
    ci = _pcov(X, Y) / math.sqrt(_pcov(X, X) * _pcov(Y, Y))
    ce = _pcov(bx, by) / math.sqrt(_pcov(bx, bx) * _pcov(by, by))
    assert d.corr_individual == pytest.approx(ci, abs=1e-12)
    assert d.corr_ecological == pytest.approx(ce, abs=1e-12)
    assert d.within_share_of_cov == pytest.approx(d.cov["within"] / d.cov["individual"], abs=1e-12)


def test_zero_within_covariance_bounds_the_individual_correlation():
    # within part of y orthogonal to the within part of x in every group
    g = ["a"] * 4 + ["b"] * 4
    x = [0.0, 1.0, 2.0, 3.0, 5.0, 4.0, 7.0, 6.0]
    wy = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]  # orthogonal to centred x within each group
    y = [(5.0 if gg == "a" else 1.0) + w for gg, w in zip(g, wy)]
    d = ecological_decompose(x, y, g)
    assert d.bound_applies
    assert d.corr_individual**2 <= d.corr_ecological**2 + 1e-12


def test_robinson_reversal():
    d = ecological_decompose([0, 2, 1, 3], [1, 3, 0, 2], ["a", "a", "b", "b"])
    assert d.corr_individual == pytest.approx(0.75 / math.sqrt(1.25 * 1.25), abs=1e-12)
    assert d.corr_ecological == pytest.approx(-1.0, abs=1e-12)
    assert d.sign_reversed and not d.bound_applies
    with pytest.raises(ValueError, match="equal length"):
        ecological_decompose([1, 2, 3], [1, 2], ["a", "a", "b"])
