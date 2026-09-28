"""sttools against vegan (mantel), MASS (kde2d), stats (ccf, prcomp) and hand-derived Knox, near-repeat, Rossmo and aoristic values."""

import math

import pytest

from morie.fn._rng import random_uniform
from morie.fn.sttools import (
    aoristic_weights,
    eof_analysis,
    geographic_profile,
    kde2d,
    knox_space_time,
    mantel_matrix_test,
    near_repeat_table,
    sample_cross_correlation,
)

REF = {
    "mantel": 0.84975606214384258,
    "mantel_s": 0.84906922938987106,
    "ccf": [
        0.30996448894970935,
        0.54060558151511684,
        0.68766235682743715,
        0.81446851400761699,
        0.86529942044765218,
        0.75877331200730647,
        0.58986968784148897,
        0.32831354596829548,
        0.022779862564475485,
        -0.21892762693443929,
        -0.40471547462860946,
    ],
    "kde_row": [
        0.040344584223725242,
        0.064477214953963871,
        0.078502740644222094,
        0.055707295052120125,
        0.043940821620001944,
        0.014439222430394085,
    ],
    "explained": [
        0.66114372745990901,
        0.33018920546429992,
        0.0027241198051223899,
        0.0025350809985767099,
        0.0018090549498086536,
        0.0015988113222832989,
    ],
}

_u = [float(v) for v in random_uniform(400, seed=101, stream=0)]
P = [(5 * _u[i], 5 * _u[30 + i]) for i in range(30)]
A = [[math.dist(a, b) for b in P] for a in P]
B = [[abs(_u[60 + i] - _u[60 + j]) + 0.3 * A[i][j] for j in range(30)] for i in range(30)]
X = [math.sin(i / 3) + _u[100 + i] for i in range(40)]
Y = [math.sin((i - 2) / 3) + _u[150 + i] for i in range(40)]
Z = [[math.sin(t / 4 + s) + 0.3 * _u[200 + (t * 7 + s) % 190] for s in range(6)] for t in range(20)]
KX = [_u[i] * 4 for i in range(50)]
KY = [_u[50 + i] * 3 + _u[i] for i in range(50)]


def test_equal_vegan_mass_stats():
    assert mantel_matrix_test(A, B, nsim=0).statistic == pytest.approx(REF["mantel"], abs=1e-14)
    assert mantel_matrix_test(A, B, method="spearman", nsim=0).statistic == pytest.approx(REF["mantel_s"], abs=1e-14)
    assert sample_cross_correlation(X, Y, 5).acf == pytest.approx(REF["ccf"], abs=1e-14)
    assert kde2d(KX, KY, n=6).z[2] == pytest.approx(REF["kde_row"], abs=1e-14)
    assert eof_analysis(Z).explained == pytest.approx(REF["explained"][:6], abs=1e-14)


def test_knox_near_repeat_by_hand():
    k = knox_space_time([(0, 0), (0.5, 0), (5, 5), (5.2, 5)], [1, 2, 10, 11], 1.0, 2.0, nsim=19)
    assert (k.observed, k.n_space, k.n_time) == (2, 2, 2)
    assert k.expected == pytest.approx(2 * 2 / 6)
    assert k.poisson_p == pytest.approx(1 - math.exp(-2 / 3) * (1 + 2 / 3), abs=1e-14)
    assert 0 < k.mc_p <= 1
    t = near_repeat_table([(0, 0), (0.5, 0), (5, 5), (5.2, 5)], [1, 2, 10, 11], [1, 10], [2, 20], nsim=9)
    assert t.observed == [[2, 0], [0, 4]]
    # every permutation keeps the 6 pairs, so the expected table sums to 6
    assert sum(v for r in t.expected for v in r) == pytest.approx(6.0)


def test_rossmo_and_aoristic_by_hand():
    B_, f = 1.5, 1.2
    a = 2 * B_ ** (0.0) / (2 * B_ - 1) ** f
    b = 1 / 10**f + 1 / 8**f
    assert geographic_profile([(0, 0), (2, 0)], [(1, 0), (5, 5)], B=1.5) == pytest.approx([a / (a + b), b / (a + b)])
    r = aoristic_weights([0.5, 2.0], [2.5, 2.0], [0, 1, 2, 3])
    assert r.weights == [[0.25, 0.5, 0.25], [0.0, 0.0, 1.0]] and r.totals == [0.25, 0.5, 1.25]
    with pytest.raises(ValueError):
        mantel_matrix_test(A, B, method="kendall", nsim=0)
