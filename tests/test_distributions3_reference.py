"""GHS, Bingham and BB1 densities: closed forms, exact normalisations and finite-difference checks."""

import math

from morie.fn._distcore import gauss_legendre
from morie.fn._rng import random_uniform
from morie.fn.binghamdens import binghamdens
from morie.fn.copuladens import copuladens
from morie.fn.ghsecant import ghsecant

A3 = [[2.0, 0.5, 0.1], [0.5, -1.0, 0.3], [0.1, 0.3, 0.4]]
A2 = [[1.5, 0.4], [0.4, -0.7]]


def close(a, b, tol=1e-12):
    return all(abs(x - y) <= tol * max(1.0, abs(y)) for x, y in zip(a, b))


def test_ghs_special_cases_are_sech_and_logistic():
    xs = [-2.0, -0.3, 0.0, 1.1]
    got = ghsecant(xs, t=-math.pi / 2)["pdf"]
    assert close(got, [0.5 / math.cosh(math.pi * x / 2) for x in xs], 1e-14)
    s = math.sqrt(3) / math.pi
    got = ghsecant(xs, t=0.0)
    assert close(got["pdf"], [math.exp(-x / s) / (s * (1 + math.exp(-x / s)) ** 2) for x in xs], 1e-14)
    assert close(got["cdf"], [1 / (1 + math.exp(-x / s)) for x in xs], 1e-14)
    assert ghsecant(0.0, t=-math.pi / 2)["pdf"] == 0.5  # docstring example


def test_ghs_is_standardised_and_cdf_quantile_consistent():
    for t in (-2.5, -1.0, 0.0, 0.7, 3.0):
        f = lambda v, t=t: ghsecant(v, t=t)["pdf"]  # noqa: E731
        m = [gauss_legendre(lambda v, k=k: v**k * f(v), -60.0, 60.0, n=600) for k in (0, 1, 2)]
        assert close(m, [1.0, 0.0, 1.0], 1e-11)
        xs = [-1.3, 0.2, 2.1]
        res = ghsecant(xs, t=t, loc=0.5, scale=2.0)
        num = [gauss_legendre(lambda v: ghsecant(v, t=t, loc=0.5, scale=2.0)["pdf"], -150.0, x, n=800) for x in xs]
        assert close(res["cdf"], num, 1e-11)
        assert close(ghsecant(p=res["cdf"], t=t, loc=0.5, scale=2.0)["quantile"], xs, 1e-11)
        assert close(res["logpdf"], [math.log(d) for d in res["pdf"]], 1e-14)


def test_ghs_pinned_values_and_draws():
    ref = {
        "-2.5": [
            [0.02694960625408328, 0.23188339487486, 0.4996082829426766, 0.0906663762630237],
            [0.0291581653477348, 0.18985533700056711, 0.44944298630982954, 0.914301276811783],
        ],
        "-0.5": [
            [0.035028589685646594, 0.2591039271392708, 0.3499487736787849, 0.1350979274356413],
            [0.026054301926301537, 0.2454023220580126, 0.46488713985249497, 0.8908018037655583],
        ],
        "0.7": [
            [0.035482682360713995, 0.2596104316587597, 0.3418988258768606, 0.13877582571270963],
            [0.02557487771749023, 0.24917814533003046, 0.4657067398556496, 0.8890852303807263],
        ],
        "3.0": [
            [0.03768734752145766, 0.2573851237703502, 0.29054540782720145, 0.16841147463571185],
            [0.02078553651573445, 0.2759204751362695, 0.4709127808044994, 0.8758494902410078],
        ],
    }
    for t in (-2.5, -0.5, 0.7, 3.0):
        res = ghsecant([-2.2, -0.4, 0.3, 1.9], t=t, loc=0.4, scale=1.3)
        assert close(res["pdf"], ref[str(t)][0]) and close(res["cdf"], ref[str(t)][1])
    rnd = ghsecant(t=1.1, n=3, seed=5)["random"]
    assert close(rnd, ghsecant(p=random_uniform(3, seed=5, stream=0), t=1.1)["quantile"], 1e-15)
    assert close(rnd, [0.6714564047525892, -1.4356729899412328, 2.232557010463369])


def test_bingham_normalisation_is_exact():
    assert abs(binghamdens([1.0, 0.0, 0.0], [[0.0] * 3] * 3)["pdf"] - 1 / (4 * math.pi)) < 1e-15
    assert abs(binghamdens([0.0, 1.0], [[0.0] * 2] * 2)["pdf"] - 1 / (2 * math.pi)) < 1e-15
    n = 400  # periodic trapezoid on the circle is spectrally accurate
    pts = [[math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)] for k in range(n)]
    assert abs(sum(binghamdens(pts, A2)["pdf"]) * 2 * math.pi / n - 1) < 1e-13
    # sphere: Gauss-Legendre in z = cos(theta) times periodic trapezoid in phi
    tot = 0.0
    for z0 in range(40):
        a, b = -1 + z0 / 20, -1 + (z0 + 1) / 20
        for zz, wz in (
            (0.5 * (a + b) + 0.5 * (b - a) * q, 0.5 * (b - a) * w)
            for q, w in ((-0.7745966692414834, 5 / 9), (0.0, 8 / 9), (0.7745966692414834, 5 / 9))
        ):
            r = math.sqrt(1 - zz * zz)
            pts = [[r * math.cos(2 * math.pi * k / 64), r * math.sin(2 * math.pi * k / 64), zz] for k in range(64)]
            tot += wz * sum(binghamdens(pts, A3)["pdf"]) * 2 * math.pi / 64
    assert abs(tot - 1) < 1e-9


def test_bingham_invariances_and_pinned_values():
    shifted = [[A3[i][j] + (0.7 if i == j else 0.0) for j in range(3)] for i in range(3)]
    assert (
        abs(
            binghamdens([1.0, 0.0, 0.0], shifted)["log_normalizer"]
            - binghamdens([1.0, 0.0, 0.0], A3)["log_normalizer"]
            - 0.7
        )
        < 1e-12
    )
    x = [0.6, 0.0, 0.8]
    assert abs(binghamdens(x, A3)["pdf"] - binghamdens([-v for v in x], A3)["pdf"]) < 1e-15
    res = binghamdens([[0.6, 0.0, 0.8], [0.0, -1.0, 0.0]], A3)
    assert close(res["logpdf"], [-2.259381723397146, -4.331381723397146])
    assert close([binghamdens([[0.6, 0.8]], A2)["logpdf"][0]], [-2.0788510281267074])


def test_bb1_matches_mixed_partial_and_limits():
    th, de = 0.8, 1.7

    def C(u, v):
        return (1 + ((u**-th - 1) ** de + (v**-th - 1) ** de) ** (1 / de)) ** (-1 / th)

    h = 1e-4
    for u, v in ((0.3, 0.7), (0.1, 0.15), (0.9, 0.6), (0.5, 0.5)):
        fd = (C(u + h, v + h) - C(u + h, v - h) - C(u - h, v + h) + C(u - h, v - h)) / (4 * h * h)
        assert abs(copuladens(u, v, "bb1", th, delta=de)["density"] - fd) < 1e-6
    assert (
        abs(copuladens(0.3, 0.7, "bb1", 1.3, delta=1.0)["density"] - copuladens(0.3, 0.7, "clayton", 1.3)["density"])
        < 1e-14
    )
    assert (
        abs(copuladens(0.3, 0.7, "bb1", 1e-7, delta=2.0)["density"] - copuladens(0.3, 0.7, "gumbel", 2.0)["density"])
        < 1e-6
    )
    got = [copuladens(u, v, "bb1", 0.8, delta=1.7)["density"] for u, v in ((0.3, 0.7), (0.1, 0.15), (0.9, 0.6))]
    assert close(got, [0.5630155340857571, 3.243167350850808, 0.754921952483426])
