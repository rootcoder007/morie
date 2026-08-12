"""Tests for causrddc (Calonico, Cattaneo & Titiunik 2014)."""

import math

from morie.fn.causrddc import (causrddc, kernel_constants, local_poly_weights,
                               rd_bandwidth, rdrobust)


def _lcg(seed):
    state = [seed]

    def f():
        state[0] = (1103515245 * state[0] + 12345) % (1 << 31)
        return state[0] / float(1 << 31)
    return f


def _normal(rnd):
    u1 = max(rnd(), 1e-12)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * rnd())


def _make(n, seed, tau=1.0, noise=0.3, cubic=False):
    rnd = _lcg(seed)
    x, y = [], []
    for _ in range(n):
        xi = 2.0 * rnd() - 1.0
        mu = 0.5 * xi + 0.8 * xi * xi + (0.4 * xi ** 3 if cubic else 0.0)
        x.append(xi)
        y.append(mu + (tau if xi >= 0 else 0.0) + noise * _normal(rnd))
    return x, y


def test_uniform_kernel_constants_are_exact():
    G, th, P = kernel_constants(3, 4, "uniform")
    for a in range(4):
        assert abs(th[a] - 1.0 / (4 + a + 1)) < 1e-9
        for b in range(4):
            assert abs(G[a][b] - 1.0 / (a + b + 1)) < 1e-9
            assert abs(P[a][b] - G[a][b]) < 1e-12


def test_local_polynomial_weights_are_exact_on_polynomials():
    x, _ = _make(600, 11)
    for nu, p in ((0, 1), (0, 2), (1, 2)):
        w, _ = local_poly_weights(x, 0.5, p, nu, "triangular", +1)
        for t in range(p + 1):
            got = sum(w[i] * x[i] ** t / math.factorial(t)
                      for i in range(len(x)))
            assert abs(got - (1.0 if t == nu else 0.0)) < 1e-8
        assert all(w[i] == 0.0 for i in range(len(x)) if x[i] < 0.0)


def test_remark_7_identity():
    """h = b makes the bias-corrected estimator the order p+1 estimator."""
    x, y = _make(800, 7)
    for p in (1, 2):
        lo = causrddc(y, x, p=p, h=0.5, b=0.5)
        hi = causrddc(y, x, p=p + 1, h=0.5)
        assert abs(lo["bias_corrected"] - hi["estimate"]) < 1e-11
        assert abs(lo["se_robust"] - hi["se_conventional"]) < 1e-12
        assert abs(lo["estimate"] - hi["estimate"]) > 1e-6


def test_exact_recovery_on_noiseless_polynomial_data():
    x, y = _make(400, 23, tau=1.5, noise=0.0, cubic=True)
    assert abs(causrddc(y, x, p=3, h=0.6)["estimate"] - 1.5) < 1e-8
    assert abs(causrddc(y, x, p=2, h=0.6)["estimate"] - 1.5) > 1e-6
    corrected = causrddc(y, x, p=2, q=3, h=0.6, b=0.6)
    assert abs(corrected["bias_corrected"] - 1.5) < 1e-8


def test_robust_interval_is_wider_and_covers_at_least_as_often():
    cov_c = cov_r = 0
    wider = 0
    reps = 40
    for rep in range(reps):
        x, y = _make(500, 1000 + 37 * rep, tau=1.0, noise=0.25)
        r = causrddc(y, x, p=1)
        lo, hi = r["ci_conventional"]
        cov_c += lo <= 1.0 <= hi
        lo2, hi2 = r["ci_robust"]
        cov_r += lo2 <= 1.0 <= hi2
        wider += (hi2 - lo2) > (hi - lo)
    assert wider == reps
    assert cov_r >= cov_c


def test_bandwidth_formula():
    x, y = _make(1500, 5150, tau=1.0, noise=0.25)
    for nu, p in ((0, 1), (0, 2)):
        bw = rd_bandwidth(x, y, nu, p)
        assert abs(bw["h_unclamped"] -
                   bw["C"] * len(x) ** (-1.0 / (2.0 * p + 3.0))) < 1e-12
        want = ((1.0 + 2.0 * nu) * bw["V"] /
                (2.0 * (p + 1.0 - nu) * bw["B"] ** 2)) ** (1.0 /
                                                           (2.0 * p + 3.0))
        assert abs(bw["C"] - want) < 1e-12
        assert bw["h"] <= max(abs(v) for v in x) + 1e-12


def test_fuzzy_with_perfect_compliance_is_sharp():
    x, y = _make(800, 7)
    t = [1.0 if v >= 0 else 0.0 for v in x]
    sharp = causrddc(y, x, p=1, h=0.5, b=0.5)
    fuzzy = causrddc(y, x, t, p=1, h=0.5, b=0.5)
    assert abs(fuzzy["estimate"] - sharp["estimate"]) < 1e-9
    assert abs(fuzzy["bias_corrected"] - sharp["bias_corrected"]) < 1e-9
    assert fuzzy["fuzzy"]


def test_variance_routes_agree_on_the_point_estimate():
    x, y = _make(800, 7)
    nn = causrddc(y, x, p=1, h=0.5, b=0.5, vce="nn")
    hc = causrddc(y, x, p=1, h=0.5, b=0.5, vce="hc")
    assert abs(nn["estimate"] - hc["estimate"]) < 1e-12
    assert 0.5 < nn["se_robust"] / hc["se_robust"] < 2.0


def test_validation():
    x, y = _make(400, 3)
    for call in (lambda: causrddc(y, x, p=2, q=2),
                 lambda: causrddc(y, x, nu=2, p=1),
                 lambda: causrddc(y, x, h=-1.0),
                 lambda: causrddc(y, x, kernel="gaussian"),
                 lambda: causrddc(y, x, vce="boot"),
                 lambda: causrddc(y, x, alpha=1.5),
                 lambda: causrddc(y[:-1], x),
                 lambda: causrddc(y, x, [1.0] * len(x), h=0.5, b=0.5)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert rdrobust is causrddc
