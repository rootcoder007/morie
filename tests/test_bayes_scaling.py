"""Bayesian spatial-scaling samplers (the R arm's models): helpers against
closed forms, and small recovery runs."""

import math
from statistics import NormalDist

import pytest

from morie import _bayes_scaling as B
from morie.fn import _array_core as np


def test_log_ncdf_matches_the_cdf_and_its_tail_expansion():
    # R's pnorm(x, log.p = TRUE); NormalDist().cdf loses digits in the lower tail
    r_vals = {
        -5.0: -15.064998393988725,
        -1.0: -1.8410216450092636,
        0.0: -0.69314718055994529,
        2.0: -0.023012909328963493,
    }
    for x, v in r_vals.items():
        assert B._log_ncdf(x) == pytest.approx(v, abs=1e-12)
    # Mills-ratio expansion far in the tail: log Phi(-40) = -800 - log 40 - log sqrt(2 pi) + log(1 - 1/1600 + ...)
    x = -40.0
    expect = -0.5 * x * x - math.log(-x) - 0.5 * math.log(2 * math.pi) + math.log(1 - 1 / x**2 + 3 / x**4)
    assert B._log_ncdf(x) == pytest.approx(expect, rel=1e-12)


def test_regularised_gamma_against_closed_forms():
    # P(1, x) = 1 - exp(-x); P(2, x) = 1 - (1 + x) exp(-x)
    for x in (0.3, 1.0, 4.0, 12.0):
        assert B._pgamma(x, 1.0) == pytest.approx(1 - math.exp(-x), abs=1e-13)
        assert B._pgamma(x, 2.0) == pytest.approx(1 - (1 + x) * math.exp(-x), abs=1e-13)


def test_truncated_normal_stays_in_bounds_with_the_right_mean():
    rng = np.random.default_rng(1)
    xs = [B._rtnorm(rng, 0.3, 1.2, -0.5, 2.0) for _ in range(20000)]
    assert min(xs) >= -0.5 and max(xs) <= 2.0
    a, b = (-0.5 - 0.3) / 1.2, (2.0 - 0.3) / 1.2
    nd = NormalDist()
    m = 0.3 + 1.2 * (nd.pdf(a) - nd.pdf(b)) / (nd.cdf(b) - nd.cdf(a))
    assert abs(sum(xs) / len(xs) - m) < 4 * 1.2 / math.sqrt(20000)


def test_tau_draw_is_the_gamma_conditional_truncated_at_ten():
    rng = np.random.default_rng(2)
    draws = [B._bp_tau(rng, 200, 400.0) for _ in range(3000)]
    # Gamma(101, rate 200): mean 0.505, far below the bound
    assert abs(sum(draws) / len(draws) - 101 / 200) < 4 * math.sqrt(101) / 200 / math.sqrt(3000)
    assert all(0 < B._bp_tau(rng, 200, 1.0) <= 10 for _ in range(200))


def test_slice_sampler_draws_from_its_target():
    rng = np.random.default_rng(3)
    x = [0.0]
    out = []
    for _ in range(4000):
        x = B._slice_vec(rng, lambda v: [-0.5 * ((t - 1.5) / 0.7) ** 2 for t in v], x, 8.0)
        out.append(x[0])
    m = sum(out) / len(out)
    sd = math.sqrt(sum((t - m) ** 2 for t in out) / (len(out) - 1))
    assert abs(m - 1.5) < 0.06
    assert abs(sd - 0.7) < 0.05


def test_bayesian_am_recovers_standardised_positions():
    rng = np.random.default_rng(11)
    zt = [-1.4, -0.6, 0.0, 0.5, 1.5]
    Z = []
    for _ in range(60):
        a = float(rng.normal(0, 0.8))
        b = float(rng.uniform(0.4, 1.6))
        Z.append([a + b * z + float(rng.normal(0, 0.3)) for z in zt])
    f = B.bayes_am(Z, n_samples=150, burn_in=100, seed=1)
    m = sum(zt) / 5
    sd = math.sqrt(sum((z - m) ** 2 for z in zt) / 4)
    target = [(z - m) / sd for z in zt]
    assert max(abs(f["zeta_mean"][j] - target[j]) for j in range(5)) < 0.08
    for d in f["draws"]:
        assert sum(d) / 5 == pytest.approx(0.0, abs=1e-12)
        assert d[0] < 0  # polarity stimulus on the left
    with pytest.raises(ValueError, match="two stimuli"):
        B.bayes_am([[1.0], [2.0]])


def test_bayesian_mds_recovers_distances():
    rng = np.random.default_rng(4)
    X = [[float(rng.normal()), float(rng.normal())] for _ in range(7)]
    D = [[math.dist(X[i], X[j]) for j in range(7)] for i in range(7)]
    f = B.bayes_mds(D, 2, 150, 100, seed=1)
    pairs = [(i, j) for i in range(7) for j in range(i + 1, 7)]
    a = [f["distance_mean"][i][j] for i, j in pairs]
    b = [D[i][j] for i, j in pairs]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cor = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(
        sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)
    )
    assert cor > 0.98
    assert 0 < f["tau"] <= 10
    with pytest.raises(ValueError, match="square"):
        B.bayes_mds([[0.0, 1.0, 2.0]])


def test_ordinal_irt_has_ordered_item_cutpoints():
    rng = np.random.default_rng(7)
    th = [float(rng.normal()) for _ in range(60)]
    Y = []
    for t in th:
        row = []
        for _ in range(10):
            e = 1.2 * t + float(rng.normal())
            row.append(1 if e < -0.5 else (2 if e < 0.5 else 3))
        Y.append(row)
    f = B.ordinal_irt(Y, 1, 150, 100, seed=2)
    ip = [r[0] for r in f["ideal_points"]]
    m1, m2 = sum(ip) / 60, sum(th) / 60
    cor = sum((a - m1) * (b - m2) for a, b in zip(ip, th)) / math.sqrt(
        sum((a - m1) ** 2 for a in ip) * sum((b - m2) ** 2 for b in th)
    )
    assert cor > 0.85
    assert f["discrimination"][0][0] > 0
    for g in f["cutpoints"]:
        assert g[0] == 0.0 and all(g[k] < g[k + 1] for k in range(len(g) - 1))


def test_alpha_nominate_probability_is_the_utility_mixture():
    beta, alpha, dy, dn = 5.0, 0.4, 0.3, 0.9
    q = -0.5 * beta * 0.25 * (dy - dn)
    g = beta * (math.exp(-0.125 * dy) - math.exp(-0.125 * dn))
    u = q + alpha * (g - q)
    assert B._anom_ll(1.0, dy, dn, beta, alpha) == pytest.approx(math.log(NormalDist().cdf(u)), abs=1e-12)
    assert B._anom_ll(0.0, dy, dn, beta, alpha) == pytest.approx(math.log(NormalDist().cdf(-u)), abs=1e-12)


def test_alpha_nominate_small_run_keeps_its_identification():
    rng = np.random.default_rng(5)
    x = sorted(float(rng.uniform(-1, 1)) for _ in range(12))
    mids = [float(rng.uniform(-0.7, 0.7)) for _ in range(25)]
    V = [[1.0 if (xi > mj) == (float(rng.uniform()) > 0.1) else 0.0 for mj in mids] for xi in x]
    f = B.alpha_nominate(V, 1, 20, 10, seed=3, minvotes=10, polarity=11)
    assert 0.0 <= f["alpha"] <= 1.0 and f["beta"] > 0
    for d in f["draws"]["X"]:
        assert sum(r[0] for r in d) / len(d) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="1 \\(yea\\)"):
        B.alpha_nominate([[2.0, 0.0], [1.0, 0.0]])


def test_dynamic_irt_reflects_onto_the_anchor():
    rng = np.random.default_rng(9)
    th = [float(rng.normal()) for _ in range(15)]
    per = [1] * 10 + [2] * 10
    V = [[1.0 if (t * 1.5 - 0.2 * k % 1 + float(rng.normal())) > 0 else 0.0 for k in range(20)] for t in th]
    anchor = max(range(15), key=lambda i: th[i])
    f = B.dynamic_irt(V, per, 60, 40, seed=1, anchor=anchor)
    assert len(f["theta"]) == 15 and len(f["theta"][0]) == 2
    assert sum(f["theta"][anchor]) > 0
    with pytest.raises(ValueError, match="one period per roll call"):
        B.dynamic_irt(V, [1, 2])
