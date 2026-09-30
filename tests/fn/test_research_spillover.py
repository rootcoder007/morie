import itertools
import math

import pytest

from morie.fn._rng import random_uniform
from morie.fn.research_spillover import (
    spillover_effects,
    spillover_exposure,
    spillover_exposure_probs,
    spillover_ht,
    spillover_ht_variance,
)

E = [(i, i + 1) for i in range(1, 8)] + [(1, 5)]


def _exposure(tr):
    nb = [0] * len(tr)
    for a, b in E:
        nb[a - 1] += tr[b - 1]
        nb[b - 1] += tr[a - 1]
    return [2 if t else (1 if k else 0) for t, k in zip(tr, nb)]


def test_exposure_levels():
    tr = [1, 0, 0, 0, 0, 0, 1, 0]
    e = spillover_exposure(tr, E)
    assert e["exposure"] == _exposure(tr) == [2, 1, 0, 0, 1, 1, 2, 1]
    assert e["treated_neighbours"] == [0, 1, 0, 0, 1, 1, 0, 1]
    with pytest.raises(ValueError, match="1..length"):
        spillover_exposure([1, 0], [(1, 3)])


def test_effects_decomposition_and_pooling_bias():
    n = 60
    ex = [(i * 7) % 3 for i in range(n)]
    st = [f"s{i % 4}" for i in range(n)]
    y = [5 - 1.1 * (e == 2) - 0.4 * (e == 1) + 0.3 * (i % 4) + 0.05 * math.sin(i) for i, e in enumerate(ex)]
    w = [1 + (i % 5) for i in range(n)]
    r = spillover_effects(y, ex, st, w)
    tw = sum(w)
    for lv, key in enumerate(("Y0", "Y1", "Y2")):
        want = 0.0
        for s in dict.fromkeys(st):
            idx = [i for i in range(n) if st[i] == s and ex[i] == lv]
            mk = sum(w[i] for i in range(n) if st[i] == s) / tw
            want += mk * sum(w[i] * y[i] for i in idx) / sum(w[i] for i in idx)
        assert r.means[key] == pytest.approx(want, abs=1e-12)
    assert r.total == pytest.approx(r.spillover + r.direct, abs=1e-12)
    assert r.positivity
    s0 = [i for i in range(n) if st[i] == "s0" and ex[i] in (0, 1)]
    pooled = sum(w[i] * y[i] for i in s0) / sum(w[i] for i in s0)
    assert r.pooling_bias["by_stratum"]["s0"] == pytest.approx(pooled - r.cell_means["s0"][0], abs=1e-12)


def test_exact_probabilities_enumerate_every_assignment():
    pr = spillover_exposure_probs(8, E, 3, joint=True)
    count = [[0, 0, 0] for _ in range(8)]
    j11 = 0
    combos = list(itertools.combinations(range(8), 3))
    for c in combos:
        ex = _exposure([1 if i in c else 0 for i in range(8)])
        for i in range(8):
            count[i][ex[i]] += 1
        j11 += ex[0] == 1 and ex[3] == 1
    assert pr["marginal"] == [[v / len(combos) for v in row] for row in count]
    assert pr["joint"][1][0][3] == pytest.approx(j11 / len(combos), abs=1e-15)
    assert all(abs(sum(row) - 1) < 1e-12 for row in pr["marginal"])


def test_monte_carlo_probabilities_replay_the_philox_draws():
    mc = spillover_exposure_probs(8, E, 3, n_draws=50, exact_max=10, seed=9)
    u = random_uniform(400, seed=9, stream=0)
    count = [[0, 0, 0] for _ in range(8)]
    for k in range(50):
        uk = [u[k * 8 + i] for i in range(8)]
        trt = sorted(range(8), key=lambda i: uk[i])[:3]
        ex = _exposure([1 if i in trt else 0 for i in range(8)])
        for i in range(8):
            count[i][ex[i]] += 1
    assert mc == [[v / 50 for v in row] for row in count]


def test_horvitz_thompson_is_unbiased_over_the_design():
    pr = spillover_exposure_probs(8, E, 3, joint=True)
    y0 = [3 + 0.5 * math.cos(i) for i in range(8)]
    ypot = {0: y0, 1: [v - 0.3 for v in y0], 2: [v - 1 for v in y0]}
    combos = list(itertools.combinations(range(8), 3))
    tots = [0.0, 0.0, 0.0]
    for c in combos:
        ex = _exposure([1 if i in c else 0 for i in range(8)])
        y = [ypot[e][i] for i, e in enumerate(ex)]
        r = spillover_ht(y, ex, pr["marginal"], pr["joint"] if c == combos[0] else None)
        for lv, k in enumerate(("T0", "T1", "T2")):
            tots[lv] += r.totals[k] / len(combos)
    for lv in range(3):
        assert tots[lv] == pytest.approx(sum(ypot[lv]), abs=1e-9)  # ht_unbiased


def test_design_variance_forms():
    pr = spillover_exposure_probs(8, E, 3, joint=True)
    yp = [2 + math.sin(i) for i in range(8)]
    for lv in range(3):
        pi = [p[lv] for p in pr["marginal"]]
        J = pr["joint"][lv]
        c = [a / b for a, b in zip(yp, pi)]
        ht = sum((J[i][j] - pi[i] * pi[j]) * c[i] * c[j] for i in range(8) for j in range(8))
        syg = 0.5 * sum((pi[i] * pi[j] - J[i][j]) * (c[i] - c[j]) ** 2 for i in range(8) for j in range(8))
        assert spillover_ht_variance(yp, pi, J) == pytest.approx(ht, abs=1e-9)
        v = spillover_ht_variance(yp, pi, J, "both")
        assert v.syg == pytest.approx(syg, abs=1e-9)
        if v.fixed_size:
            assert v.ht == pytest.approx(v.syg, abs=1e-9)  # syg_eq_ht
    # the treated level of a completely randomised design is fixed-size: SYG = HT
    pi2 = [p[2] for p in pr["marginal"]]
    v2 = spillover_ht_variance(yp, pi2, pr["joint"][2], "both")
    assert v2.fixed_size and v2.ht == pytest.approx(v2.syg, abs=1e-9)
    assert spillover_ht_variance(pi2, pi2, pr["joint"][2], "syg") == pytest.approx(0.0, abs=1e-12)
