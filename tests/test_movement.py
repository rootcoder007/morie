import math

from morie.fn._qpcore import ssum
from morie.fn.movement import (
    brownian_bridge_ud,
    planar_brownian_motion,
    correlated_random_walk,
    crw_msd,
    lattice_random_walk,
    site_percolation,
)


def test_lattice_and_brownian_msd():
    r = lattice_random_walk(20, nwalk=400, seed=5)
    assert all(
        abs(abs(p[t + 1][0] - p[t][0]) + abs(p[t + 1][1] - p[t][1]) - 1) == 0 for p in r.paths[:5] for t in range(20)
    )
    se = math.sqrt(ssum((q[20][0] ** 2 + q[20][1] ** 2 - r.msd[20]) ** 2 for q in r.paths) / 399 / 400)
    assert abs(r.msd[20] - 20) / se < 4
    b = planar_brownian_motion(10, 0.5, sigma=2.0, nwalk=400, seed=6)
    d = [q[10][0] ** 2 + q[10][1] ** 2 for q in b.paths]
    se = math.sqrt(ssum((v - b.msd[10]) ** 2 for v in d) / 399 / 400)
    assert abs(b.msd[10] - 2 * 4 * 5) / se < 4


def test_correlated_walk_msd_matches_theory():
    r = correlated_random_walk(15, kappa=3.0, nwalk=400, seed=7)
    d = [q[15][0] ** 2 + q[15][1] ** 2 for q in r.paths]
    se = math.sqrt(ssum((v - r.msd[15]) ** 2 for v in d) / 399 / 400)
    assert abs(r.msd[15] - r.theory[15]) / se < 4
    assert crw_msd(1, kappa=2.0)[1] == 1.0


def test_bbmm_and_percolation():
    r = brownian_bridge_ud([(0, 0), (4, 0), (4, 4)], [0, 2, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], sig1=0.7, sig2=0.2)
    assert abs(ssum(v for row in r.ud for v in row) - 1) < 1e-12 and r.ud[0][2] > r.ud[4][0]
    full = site_percolation(4, 5, 1.0)
    assert full.n_clusters == [1] and full.spanning == [True]
    empty = site_percolation(4, 5, 0.0)
    assert empty.n_clusters == [0] and empty.spanning == [False]
    hi = site_percolation(20, 20, 0.75, nsim=10, seed=3)
    lo = site_percolation(20, 20, 0.35, nsim=10, seed=3)
    assert hi.spanning_probability > lo.spanning_probability
