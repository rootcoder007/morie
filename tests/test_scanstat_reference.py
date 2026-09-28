"""scanstat against smerc 1.8 (scan.zones, scan.test internals, tango.test) and SpatialEpi 1.2 (besag_newell).

Reference values printed by R on the Philox data below (both arms rebuild
it); p-values use Philox multinomial replicates, so they are checked by
arm parity and by construction, and the statistics and zones exactly.
"""

import math

import pytest

from morie.fn._rng import random_uniform
from morie.fn.scanstat import besag_newell, kulldorff_scan, scan_zones, stone_test, tango_test

N_ZONES = 331
NOC_poisson = {
    "zones": [[0, 13, 7], [9], [19], [27], [10], [15], [1]],
    "tobs": [
        16.777253511042101,
        2.9500721100645597,
        0.5284501185224042,
        0.011425474639811251,
        0.0035926777401025234,
        0.00037873690087453227,
        0,
    ],
}
NOC_binomial = {
    "zones": [[0, 13, 7], [9], [19], [27], [10], [15], [1]],
    "tobs": [
        17.076424117345596,
        2.9982699958491139,
        0.53613355572451837,
        0.011577220662729815,
        0.003639907343313098,
        0.00038368647801689804,
        0,
    ],
}
BN = {
    "m": [1, 3, 5, 3, 2, 6, 4, 3, 4, 2, 4, 5, 4, 2, 2, 2, 3, 6, 6, 3, 4, 5, 2, 5, 3, 4, 5, 4, 2, 5],
    "k": [
        26,
        45,
        22,
        23,
        21,
        21,
        24,
        44,
        28,
        22,
        25,
        22,
        24,
        40,
        21,
        21,
        22,
        32,
        26,
        25,
        24,
        30,
        21,
        26,
        25,
        25,
        25,
        25,
        22,
        28,
    ],
    "p": [
        7.7808321111594303e-05,
        6.1214315647539763e-06,
        0.91830907001148265,
        0.85912888198584625,
        0.59651359333092868,
        0.98628509955026611,
        0.95998048284289916,
        1.4828970534730423e-05,
        0.80640947235859595,
        0.09172623579654493,
        0.78245688916587974,
        0.91830907001148265,
        0.95998048284289916,
        5.2821733054209119e-07,
        0.55001560761475177,
        0.55001560761475177,
        0.81239654016010876,
        0.56737238620455954,
        0.90800388133320842,
        0.32357906889634414,
        0.95998048284289916,
        0.52378969935094388,
        0.59651359333092868,
        0.93114465723985329,
        0.32357906889634414,
        0.78245688916587974,
        0.93063064468078138,
        0.78245688916587974,
        0.09172623579654493,
        0.94440418613112587,
    ],
}
TANGO = [
    0.016761913770384464,
    0.010827027794310776,
    0.0059348859760736889,
    42.54587134873325,
    6.8846945521064962,
    3.6340093845232957e-07,
]

_u = [float(v) for v in random_uniform(200, seed=51, stream=0)]
P = [(10 * _u[i], 10 * _u[40 + i]) for i in range(30)]
POP = [round(200 + 800 * _u[80 + i]) for i in range(30)]
Y = [
    int(p * 0.01 * (3.0 if math.dist(P[i], (3, 3)) < 2.5 else 1.0) * (0.6 + 0.8 * _u[120 + i]))
    for i, p in enumerate(POP)
]


def test_zones_and_clusters_equal_smerc():
    assert len(scan_zones(P, POP)) == N_ZONES
    k = kulldorff_scan(P, Y, POP, nsim=0)
    assert k.all_zones == NOC_poisson["zones"]
    assert k.all_tobs == pytest.approx(NOC_poisson["tobs"], abs=1e-12)
    b = kulldorff_scan(P, Y, POP, kind="binomial", ubpop=0.3, nsim=0)
    assert b.all_zones == NOC_binomial["zones"]
    assert b.all_tobs == pytest.approx(NOC_binomial["tobs"], abs=1e-9)


def test_monte_carlo_pvalues_by_construction():
    k = kulldorff_scan(P, Y, POP, nsim=19, seed=3)
    # p-values are (1 + #exceedances)/(nsim + 1) and decrease with the LLR
    assert all(round(p * 20) == p * 20 for p in k.all_pvalues)
    assert k.all_pvalues[0] == pytest.approx(1 / 20)
    assert k.clusters[0]["zone"] == [0, 13, 7]
    assert k.clusters[0]["cases"] == sum(Y[j] for j in [0, 13, 7])


def test_besag_newell_and_tango_equal_references():
    r = besag_newell(P, Y, POP, 20)
    assert r.m_values == BN["m"] and r.k_values == BN["k"]
    assert r.p_values == pytest.approx(BN["p"], abs=1e-13)
    W = [[math.exp(-math.dist(a, b) / 2.0) for b in P] for a in P]
    t = tango_test(Y, POP, W)
    got = [t.tstat, t.gof, t.sa, t.tstat_chisq, t.dfc, t.pvalue_chisq]
    assert got == pytest.approx(TANGO, rel=1e-10, abs=1e-15)


def test_stone_statistic_by_hand():
    r = stone_test([4, 2, 1, 1], [1.0, 2.0, 2.0, 3.0], [0, 1, 2, 3])
    assert (r.statistic, r.k) == (4.0, 1)
    r2 = stone_test([1, 6, 1, 1], [2.0, 2.0, 2.0, 3.0], [0, 1, 2, 3], nsim=9)
    assert (r2.statistic, r2.k) == (7.0 / 4.0, 2)
    assert 0 < r2.pvalue <= 1


def test_validation():
    with pytest.raises(ValueError):
        kulldorff_scan(P, Y, POP, kind="normal", nsim=0)
