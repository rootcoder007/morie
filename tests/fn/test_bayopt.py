"""Tests for bayopt / bayoptr (Mockus 1975; Snoek et al. 2012)."""

import math

from morie.fn.bayopt import (_phi, _Phi, acquire, acquisition_gradient,
                             bayopt, gp_posterior_gradient,
                             maximise_acquisition,
                             expected_improvement, gp_posterior,
                             lower_confidence_bound, matern52,
                             probability_of_improvement,
                             squared_exponential)
from morie.fn.bayoptr import ACQUISITIONS, bayoptr, resolve_acquisition

X = [[0.0], [1.0], [2.5], [4.0]]
Y = [0.4, -0.8, 1.3, 0.1]


def _bowl(x):
    return (x[0] - 2.3) ** 2 + 0.3 * math.sin(3.0 * x[0])


def test_the_two_kernels_by_hand():
    a, b, ls = [0.3, -0.7], [1.1, 0.2], [0.5, 2.0]
    r2 = sum((a[i] - b[i]) ** 2 / ls[i] ** 2 for i in range(2))
    s = math.sqrt(5.0 * r2)
    assert abs(matern52(a, b, 1.7, ls) -
               1.7 * (1 + s + 5.0 / 3.0 * r2) * math.exp(-s)) < 1e-12
    assert abs(squared_exponential(a, b, 1.7, ls) -
               1.7 * math.exp(-0.5 * r2)) < 1e-12
    assert abs(matern52(a, a, 1.7, ls) - 1.7) < 1e-12
    assert abs(squared_exponential(a, a, 1.7, ls) - 1.7) < 1e-12


def test_matern_has_the_heavier_tail():
    assert matern52([0.0], [3.0]) > squared_exponential([0.0], [3.0])


def test_the_gp_interpolates_noiseless_data():
    post = gp_posterior(X, Y, X, noise=0.0)
    assert max(abs(post["mean"][i] - Y[i]) for i in range(4)) < 1e-7
    assert max(post["sd"]) < 1e-6


def test_far_from_the_data_it_returns_to_the_prior():
    far = gp_posterior(X, Y, [[50.0]], noise=0.0)
    assert abs(far["mean"][0] - sum(Y) / 4.0) < 1e-6
    assert abs(far["sd"][0] - 1.0) < 1e-6


def test_equations_1_to_3():
    mu, sd, best = 0.4, 0.9, 1.0
    g = (best - mu) / sd
    assert abs(probability_of_improvement(mu, sd, best) - _Phi(g)) < 1e-15
    assert abs(expected_improvement(mu, sd, best) -
               sd * (g * _Phi(g) + _phi(g))) < 1e-15
    assert abs(lower_confidence_bound(mu, sd, 2.5) -
               (mu - 2.5 * sd)) < 1e-15


def test_ei_equals_the_integral_it_is_defined_as():
    def numeric(mu, sd, best, lo=-12.0, hi=12.0, n=200000):
        tot, h = 0.0, (hi - lo) / n
        for i in range(n):
            z = lo + (i + 0.5) * h
            tot += max(0.0, best - (mu + sd * z)) * _phi(z) * h
        return tot

    for m, s, b in ((0.4, 0.9, 1.0), (2.0, 0.5, 1.0)):
        assert abs(expected_improvement(m, s, b) - numeric(m, s, b)) < 1e-4


def test_acquisition_behaviour():
    assert expected_improvement(2.0, 0.0, 1.0) == 0.0
    assert probability_of_improvement(2.0, 0.0, 1.0) == 0.0
    assert expected_improvement(1.5, 2.0, 1.0) > \
        expected_improvement(1.5, 0.5, 1.0)
    assert expected_improvement(0.4, 0.9, 1.0, xi=0.5) < \
        expected_improvement(0.4, 0.9, 1.0)
    assert abs(acquire(0.4, 0.9, 1.0, "lcb", kappa=2.5) +
               lower_confidence_bound(0.4, 0.9, 2.5)) < 1e-15


GX = [[0.2, -0.4], [1.3, 0.8], [-0.6, 1.5], [2.0, 0.1], [0.9, -1.2]]
GY = [0.3, -0.5, 1.1, 0.2, -0.9]


def _fd(fn, pt, h=1e-6):
    out = []
    for i in range(len(pt)):
        up, dn = list(pt), list(pt)
        up[i] += h
        dn[i] -= h
        out.append((fn(up) - fn(dn)) / (2 * h))
    return out


def test_the_posterior_gradients_match_finite_differences():
    for kern in ("matern52", "se"):
        for q in ([0.5, 0.5], [-0.3, 1.0], [1.8, -0.6]):
            gmu, gsd, _m, _s = gp_posterior_gradient(GX, GY, q, kern,
                                                     1.0, 0.7, 1e-8)
            nmu = _fd(lambda p: gp_posterior(GX, GY, [p], kern, 1.0, 0.7,
                                             1e-8)["mean"][0], q)
            nsd = _fd(lambda p: gp_posterior(GX, GY, [p], kern, 1.0, 0.7,
                                             1e-8)["sd"][0], q)
            assert max(abs(gmu[i] - nmu[i]) for i in range(2)) < 1e-5
            assert max(abs(gsd[i] - nsd[i]) for i in range(2)) < 1e-5


def test_the_acquisition_gradients_match_finite_differences():
    for a in ("ei", "pi", "lcb"):
        for q in ([0.5, 0.5], [-0.3, 1.0]):
            gmu, gsd, mu, sd = gp_posterior_gradient(GX, GY, q)
            g = acquisition_gradient(gmu, gsd, mu, sd, min(GY), a)
            num = _fd(lambda p: acquire(
                gp_posterior(GX, GY, [p])["mean"][0],
                gp_posterior(GX, GY, [p])["sd"][0], min(GY), a), q)
            assert max(abs(g[i] - num[i]) for i in range(2)) < 1e-5


def test_gradient_ascent_beats_sampling_the_acquisition():
    box = [(-2.0, 3.0), (-2.0, 3.0)]
    best = min(GY)
    opt = maximise_acquisition(GX, GY, best, box, "ei", n_starts=6)
    st = [999]

    def rnd():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)

    grid = []
    for _ in range(1500):
        p = [box[i][0] + rnd() * (box[i][1] - box[i][0]) for i in range(2)]
        po = gp_posterior(GX, GY, [p])
        grid.append(acquire(po["mean"][0], po["sd"][0], best, "ei"))
    assert opt["acq"] >= max(grid) - 1e-9
    assert all(box[i][0] - 1e-12 <= opt["x"][i] <= box[i][1] + 1e-12
               for i in range(2))


def test_both_inner_searches_run():
    g = bayopt(_bowl, [(-2.0, 8.0)], n_iter=20, n_init=4, seed=3)
    r = bayopt(_bowl, [(-2.0, 8.0)], n_iter=20, n_init=4, seed=3,
               inner="random")
    assert g["inner"] == "gradient" and r["inner"] == "random"
    assert g["y_best"] <= r["y_best"] + 1e-9


def test_the_loop_finds_a_known_minimum():
    res = bayopt(_bowl, [(-2.0, 8.0)], n_iter=25, n_init=4, seed=3)
    assert abs(res["x_best"][0] - 2.3) < 0.6
    assert res["n_eval"] == 29
    assert all(res["trace"][i]["best"] >= res["trace"][i + 1]["best"] - 1e-12
               for i in range(len(res["trace"]) - 1))


def test_every_acquisition_and_kernel_runs():
    for acq in ("ei", "pi", "lcb"):
        r = bayopt(_bowl, [(-2.0, 8.0)], n_iter=15, n_init=4, acq=acq,
                   seed=3)
        assert r["y_best"] <= min(r["y"][:4])
    se = bayopt(_bowl, [(-2.0, 8.0)], n_iter=15, n_init=4, kernel="se",
                seed=3)
    assert se["kernel"] == "se"


def test_two_dimensions_and_a_supplied_design():
    two = bayopt(lambda x: (x[0] - 1.0) ** 2 + (x[1] + 0.5) ** 2,
                 [(-3.0, 3.0), (-3.0, 3.0)], n_iter=25, n_init=5, seed=1)
    assert abs(two["x_best"][0] - 1.0) < 0.8
    assert abs(two["x_best"][1] + 0.5) < 0.8
    warm = bayopt(_bowl, [(-2.0, 8.0)], n_iter=5, X0=[[0.0], [4.0]],
                  seed=2)
    assert warm["n_eval"] == 7 and warm["X"][0] == [0.0]


def test_bayoptr_resolves_names_and_delegates():
    for spelling, rule in (("ei", "ei"), ("EI", "ei"),
                           ("expected_improvement", "ei"), ("pi", "pi"),
                           ("ucb", "lcb"), ("lcb", "lcb")):
        assert resolve_acquisition(spelling) == rule
    assert resolve_acquisition("ucb") == resolve_acquisition("lcb")
    assert set(ACQUISITIONS.values()) == {"ei", "pi", "lcb"}
    via = bayoptr(_bowl, [(-2.0, 8.0)], acquisition="ucb", n_iter=12,
                  n_init=4, seed=5)
    direct = bayopt(_bowl, [(-2.0, 8.0)], acq="lcb", n_iter=12, n_init=4,
                    seed=5)
    assert via["X"] == direct["X"] and via["y"] == direct["y"]
    assert via["acquisition"] == "ucb" and via["acq"] == "lcb"


def test_validation():
    for call in (lambda: bayopt(_bowl, []),
                 lambda: bayopt(_bowl, [(1.0, 1.0)]),
                 lambda: bayopt(_bowl, [(0.0, 1.0)], n_iter=0),
                 lambda: bayopt(_bowl, [(0.0, 1.0)], n_candidates=0),
                 lambda: bayopt(_bowl, [(0.0, 1.0)], n_init=1),
                 lambda: bayopt(_bowl, [(0.0, 1.0)], acq="ucb"),
                 lambda: bayopt(_bowl, [(0.0, 1.0)], kernel="rbf"),
                 lambda: bayopt(_bowl, [(0.0, 1.0)], X0=[[0.0]],
                                y0=[0.0, 1.0]),
                 lambda: gp_posterior([], [], [[0.0]]),
                 lambda: gp_posterior(X, Y[:-1], X),
                 lambda: gp_posterior(X, Y, X, noise=-1.0),
                 lambda: gp_posterior(X, Y, [[0.0, 1.0]]),
                 lambda: gp_posterior([[0.0], [0.0]], [1.0, 2.0],
                                      [[0.0]], noise=0.0),
                 lambda: matern52([0.0], [1.0], 1.0, [1.0, 2.0]),
                 lambda: matern52([0.0], [1.0], 1.0, 0.0),
                 lambda: acquire(0.0, 1.0, 1.0, acq="thompson"),
                 lambda: bayopt(_bowl, [(0.0, 1.0)], inner="grid"),
                 lambda: bayopt(_bowl, [(0.0, 1.0)], n_starts=0),
                 lambda: gp_posterior_gradient(GX, GY, [0.0]),
                 lambda: acquisition_gradient([0.0], [0.0], 0.0, 1.0,
                                              1.0, acq="thompson"),
                 lambda: resolve_acquisition("entropy_search"),
                 lambda: bayoptr(_bowl, [(0.0, 1.0)],
                                 acquisition="thompson")):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
