import math

from morie.fn._rng import random_uniform
from morie.fn.krgsys import krige
from morie.fn.krigsupport import (
    area_to_point_kriging,
    empirical_bayesian_kriging,
    empirical_variogram_bins,
    fit_variogram_wls,
    ga_kriging,
    nonstationary_kriging,
)


def _data(n=30):
    u = [float(v) for v in random_uniform(3 * n, seed=8)]
    P = [(4 * u[2 * i], 4 * u[2 * i + 1]) for i in range(n)]
    return P, [math.sin(p[0]) + math.cos(p[1]) + 0.3 * u[2 * n + i] for i, p in enumerate(P)]


def test_area_to_point_coherence():
    m = {"model": "Exp", "psill": 1.2, "range": 1.3}
    sup = [
        [(x + 0.25 + 0.5 * a, y + 0.25 + 0.5 * b) for a in range(2) for b in range(2)]
        for x in range(3)
        for y in range(3)
    ]
    vals = [1.0 + 0.3 * k + 0.1 * (k % 2) for k in range(9)]
    for k in (0, 4, 8):
        r = area_to_point_kriging(vals, sup, sup[k], m)
        assert abs(sum(r.prediction) / 4 - vals[k]) < 1e-9
        assert all(abs(sum(w) - 1) < 1e-12 for w in r.weights)


def test_constant_scale_is_stationary_exponential():
    P, z = _data()
    Q = [(1.0, 1.0), (2.5, 3.1)]
    r = nonstationary_kriging(z, P, Q, [1.3] * 30, [1.3] * 2, 1.2, nugget=0.1)
    ref = krige(z, P, Q, {"model": "Exp", "psill": 1.2, "range": 1.3, "nugget": 0.1})
    for a, b in zip(r.prediction + r.variance, ref.prediction + ref.variance):
        assert abs(a - b) < 1e-10


def test_bins_by_brute_force():
    P, z = _data(12)
    ev = empirical_variogram_bins(z, P, cutoff=2.0, width=0.5)
    pr = [(math.dist(P[a], P[b]), 0.5 * (z[a] - z[b]) ** 2) for a in range(12) for b in range(a + 1, 12)]
    k = 0
    for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)):
        sel = [(h, g) for h, g in pr if lo < h <= hi]
        if sel:
            assert ev.np[k] == len(sel)
            assert abs(ev.gamma[k] - sum(g for _, g in sel) / len(sel)) < 1e-12
            assert abs(ev.dist[k] - sum(h for h, _ in sel) / len(sel)) < 1e-12
            k += 1
    assert k == len(ev.np)


def test_fit_is_optimal_and_ga_consistent():
    P, z = _data(40)
    ev = empirical_variogram_bins(z, P)
    f = fit_variogram_wls(ev)

    def sse(c0, c1, a):
        return sum(n / h**2 * (g - c0 - c1 * (1 - math.exp(-h / a))) ** 2 for h, g, n in zip(ev.dist, ev.gamma, ev.np))

    assert abs(f.sse - sse(f.nugget, f.psill, f.range)) < 1e-10
    best = math.inf  # brute-force profile over a fine log-range grid
    for i in range(2001):
        a = math.exp(
            math.log(min(ev.dist) / 20) + (math.log(max(ev.dist) * 20) - math.log(min(ev.dist) / 20)) * i / 2000
        )
        w = [n / h**2 for h, n in zip(ev.dist, ev.np)]
        f_ = [1 - math.exp(-h / a) for h in ev.dist]
        cands = []
        s11, s12, s22 = sum(w), sum(x * y for x, y in zip(w, f_)), sum(x * y * y for x, y in zip(w, f_))
        t1, t2 = sum(x * g for x, g in zip(w, ev.gamma)), sum(x * y * g for x, y, g in zip(w, f_, ev.gamma))
        det = s11 * s22 - s12 * s12
        if det > 1e-12 * s11 * s22:
            c0, c1 = (s22 * t1 - s12 * t2) / det, (s11 * t2 - s12 * t1) / det
            if c0 >= 0 and c1 >= 0:
                cands.append((c0, c1))
        cands += [(0.0, max(t2 / s22, 0.0)), (max(t1 / s11, 0.0), 0.0)]
        best = min(best, min(sse(c0, c1, a) for c0, c1 in cands))
    assert f.sse <= best + 1e-9
    g = ga_kriging(z, P, [(1.0, 1.0)], pop=20, generations=15)
    assert g.fit["sse"] >= f.sse - 1e-9
    m = {"model": "Exp", "psill": g.fit["psill"], "range": g.fit["range"], "nugget": g.fit["nugget"]}
    assert abs(g.prediction[0] - krige(z, P, [(1.0, 1.0)], m).prediction[0]) < 1e-12


def test_ebk_mixture():
    P, z = _data(25)
    r = empirical_bayesian_kriging(z, P, [(2.0, 2.0)], nsim=6)
    assert abs(sum(r.weights) - 1) < 1e-12
    preds = [
        krige(
            z, P, [(2.0, 2.0)], {"model": "Exp", "psill": v["psill"], "range": v["range"], "nugget": v["nugget"]}
        ).prediction[0]
        for v in r.variograms
    ]
    assert abs(r.prediction[0] - sum(w * p for w, p in zip(r.weights, preds))) < 1e-12
