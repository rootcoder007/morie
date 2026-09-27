"""Point-pattern summaries against spatstat, splancs and vegan values."""

import math

from morie.fn._rng import random_uniform
from morie.fn.knox import knox_test
from morie.fn.lcfsdq import clark_evans, nn_distances
from morie.fn.mantel import mantel_test
from morie.fn.ripG import ripley_g_function
from morie.fn.ripk import ripley_k_function
from morie.fn.stk import space_time_k
from morie.fn.strmkr import strauss_process


def pattern():
    U = [float(u) for u in random_uniform(120, seed=21, stream=0)]
    return [[2 * U[2 * i], U[2 * i + 1]] for i in range(60)]


def times():
    return [10 * float(u) for u in random_uniform(60, seed=22, stream=0)]


def close(a, b, tol):
    return all(abs(u - v) <= tol for u, v in zip(a, b))


R = [0.05, 0.1, 0.15, 0.2]


def test_k_function_corrections_match_kest():
    k = ripley_k_function(pattern(), (0, 2, 0, 1), R)
    # spatstat.explore::Kest(X, r, correction = c("iso", "border", "trans")); Kest uses n(n-1)
    iso = [0.003389831, 0.018266978, 0.053770178, 0.108408179]
    trans = [0.003492317, 0.018165883, 0.052690170, 0.103961817]
    border = [0.002515723, 0.015833333, 0.048888889, 0.088]
    assert close([v * 60 / 59 for v in k["k"]], iso, 1e-9)
    assert close([v * 60 / 59 for v in k["k_trans"]], trans, 1e-9)
    assert close(k["k_border"], border, 1e-9)


def test_g_function_reduced_sample_matches_gest():
    g = ripley_g_function(pattern(), (0, 2, 0, 1), R)
    # Gest(X, r = seq(0, 0.2, by = 1e-4), correction = "rs") at 0.05, 0.1, 0.15, 0.2
    assert close(g["g_border"], [4 / 53, 0.4, 13 / 15, 1.0], 1e-9)


def test_strauss_mple_matches_ppm_on_the_same_quadrature():
    s = strauss_process(pattern(), 0.1, window=(0, 2, 0, 1), nx=12, ny=12)
    # ppm(quadscheme(X, D, method = "grid", ntile = c(12, 12)) ~ 1, Strauss(0.1), correction = "none")
    assert abs(s["beta"] - 39.230479946971) < 1e-8 and abs(s["gamma"] - 0.642950634743) < 1e-9


def test_clark_evans_matches_clarkevans():
    d = nn_distances(pattern())[0]
    c = clark_evans(d, 60, 2.0, 6.0)
    # spatstat.explore::clarkevans.test(X, correction = "none", alternative = "two.sided")
    assert abs(c["R"] - 1.1590645347059) < 1e-12 and abs(c["p"] - 0.0184185015157) < 1e-12


def test_space_time_k_matches_stkhat():
    r = space_time_k(pattern(), times(), R[:2] + [0.2, 0.3], [0.5, 1.0, 2.0, 3.0], window=(0, 2, 0, 1), tlimits=(0, 10))
    # splancs::stkhat(P, T, poly, c(0, 10), c(0.05, 0.1, 0.2, 0.3), c(0.5, 1, 2, 3))$kst[3, ]
    assert close(list(r.extra["K_st"][2]), [0.092752860557, 0.15307957042, 0.360335316901, 0.677008530433], 1e-9)
    assert close(list(r.extra["K_s"]), [0.003389831, 0.018266978, 0.108408179], 1e-9)
    D = r.extra["D"]
    assert abs(D[2][1] - (r.extra["K_st"][2][1] - r.extra["K_s"][2] * r.extra["K_t"][1])) < 1e-15


def test_mantel_and_knox_statistics():
    P, T = pattern(), times()
    D1 = [[math.dist(a, b) for b in P] for a in P]
    D2 = [[abs(s - t) for t in T] for s in T]
    # vegan::mantel(dist(P), dist(T))$statistic
    assert abs(mantel_test(D1, D2, n_perm=9, seed=1).statistic - (-0.0580122943598277)) < 1e-12
    # pairs within 0.2 in space and 2 in time
    assert knox_test(P, T, 0.2, 2.0, n_permutations=9).statistic == 27
