"""Local Geary, correlogram, k-colour join counts, bivariate LISA, Moran scatterplot.

Reference values were checked against spdep 1.4 (localC, sp.correlogram,
joincount.multi, localmoran_bv, moran.plot) to 1e-15; the cross tests in
tests/cross/test-morie_vs_spdep.R repeat that in R.
"""

import pytest

from morie.fn.gearyl import localgeary
from morie.fn.jcmult import join_count_multi
from morie.fn.jjmsta import join_count
from morie.fn.lmorbv import local_moran_bivariate
from morie.fn.morplt import moran_scatter
from morie.fn.spcorr import graph_lags, spatial_correlogram

PATH4 = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]


def _path(n):
    return [[1 if abs(i - j) == 1 else 0 for j in range(n)] for i in range(n)]


def _z(v):
    m = sum(v) / len(v)
    sd = (sum((a - m) ** 2 for a in v) / (len(v) - 1)) ** 0.5
    return [(a - m) / sd for a in v]


def test_local_geary_formula_and_docstring():
    x = [1.0, 2.0, 4.0, 8.0]
    z = _z(x)
    want = [sum(PATH4[i][j] * (z[i] - z[j]) ** 2 for j in range(4)) for i in range(4)]
    got = localgeary(x, PATH4)["local"]
    assert got == pytest.approx(want, abs=1e-12)
    assert [round(c, 6) for c in got] == [0.104348, 0.521739, 2.086957, 1.669565]
    # (2 - 1)^2 / var(x) with var(x) = 28.75 / 3
    assert got[0] == pytest.approx(3.0 / 28.75, abs=1e-12)


def test_local_geary_multivariate_is_mean_of_univariate():
    x = [1.0, 2.0, 4.0, 8.0, 3.0]
    y = [5.0, 1.0, 2.0, 2.5, 9.0]
    W = _path(5)
    a = localgeary(x, W)["local"]
    b = localgeary(y, W)["local"]
    m = localgeary([[x[i], y[i]] for i in range(5)], W)["local"]
    assert m == pytest.approx([(a[i] + b[i]) / 2 for i in range(5)], abs=1e-12)


def test_local_geary_rejects_constant():
    with pytest.raises(ValueError):
        localgeary([1.0, 1.0, 1.0], _path(3))


def test_graph_lags_on_path():
    lags = graph_lags(_path(5), 3)
    assert [j for j in range(5) if lags[1][0][j]] == [2]
    assert [j for j in range(5) if lags[2][2][j]] == []
    assert [j for j in range(5) if lags[2][1][j]] == [4]


def test_correlogram_lag1_is_morans_i():
    x = [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 8.0, 7.0]
    A = _path(8)
    r = spatial_correlogram(A, x, order=2)
    m = sum(x) / 8
    z = [v - m for v in x]
    W = [[a / sum(row) for a in row] for row in A]
    i1 = sum(z[i] * W[i][j] * z[j] for i in range(8) for j in range(8)) / sum(v * v for v in z)
    assert r.estimate[0] == pytest.approx(i1, abs=1e-12)
    assert [round(v, 6) for v in r.estimate] == [0.797619, 0.25]
    # every unit of the 8-path has a lag-2 neighbour
    assert r.expectation == pytest.approx([-1.0 / 7.0, -1.0 / 7.0], abs=1e-15)
    assert r.n_with_neighbours == [8, 8]


def test_correlogram_geary_expectation_and_errors():
    x = [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 8.0, 7.0]
    r = spatial_correlogram(_path(8), x, order=3, method="C", randomisation=False)
    assert r.expectation == [1.0, 1.0, 1.0]
    assert all(v > 0 for v in r.variance)
    with pytest.raises(ValueError):
        spatial_correlogram(_path(4), [1.0, 2.0, 4.0, 3.0], order=3)


def test_join_count_multi_counts_and_docstring():
    r = join_count_multi(list("aabbbccaa"), _path(9))
    assert r.rows == ["a:a", "b:b", "c:c", "b:a", "c:a", "c:b", "Jtot"]
    assert r.joincount == [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 3.0]


def test_join_count_multi_two_colours_matches_binary_join_count():
    lab = [1, 0, 0, 1, 1, 1, 0, 1, 0, 0]
    W = _path(10)
    r = join_count_multi(lab, W)
    b = join_count(lab, W)
    # rows 0:0, 1:1, 1:0, Jtot
    assert r.joincount[:3] == pytest.approx([b.WW, b.BB, b.BW], abs=1e-12)
    assert r.expected[:2] == pytest.approx([b.E_WW, b.E_BB], abs=1e-12)
    assert r.variance[:2] == pytest.approx([b.V_WW, b.V_BB], abs=1e-12)


def test_local_moran_bivariate_statistic_and_permutation():
    x = [1.0, 2.0, 4.0, 8.0]
    y = [2.0, 1.0, 5.0, 7.0]
    zx, zy = _z(x), _z(y)
    want = [zx[i] * sum(PATH4[i][j] * zy[j] for j in range(4)) for i in range(4)]
    r = local_moran_bivariate(x, y, PATH4, nsim=9)
    assert r.local_values == pytest.approx(want, abs=1e-12)
    assert [round(v, 6) for v in r.local_values] == [0.887109, 0.102641, 0.014663, 0.623176]
    again = local_moran_bivariate(x, y, PATH4, nsim=9)
    assert again.extra["expected"] == r.extra["expected"]
    assert all(0.0 < p <= 0.6 for p in r.extra["p_folded"])
    assert r.extra["quadrant"] == ["Low-Low", "Low-High", "High-High", "High-Low"]


def test_moran_scatter_slope_is_morans_i():
    W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    x = [1.0, 2.0, 4.0, 3.0]
    r = moran_scatter(x, W)
    m = sum(x) / 4
    z = [v - m for v in x]
    i1 = sum(z[i] * W[i][j] * z[j] for i in range(4) for j in range(4)) / sum(v * v for v in z)
    assert r.slope == pytest.approx(i1, abs=1e-12)
    assert round(r.slope, 6) == 0.3
    assert sum(r.hat) == pytest.approx(2.0, abs=1e-12)
