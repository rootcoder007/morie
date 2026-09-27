"""Multivariate and copula densities: R-package values, exact normalisations and closed forms."""

import math

from morie.fn.bvnormdist import bvnormdist
from morie.fn.bvpois import bvpois
from morie.fn.copuladens import copuladens
from morie.fn.kentdens import kentdens
from morie.fn.lkjcorr import lkjcorr
from morie.fn.multinomialdist import multinomialdist
from morie.fn.mvnormdens import mvnormdens
from morie.fn.mvskewnorm import mvskewnorm
from morie.fn.mvtdens import mvtdens
from morie.fn.vmfdens import vmfdens
from morie.fn.wishartdens import wishartdens

S = [[2.0, 0.6, 0.3], [0.6, 1.5, -0.4], [0.3, -0.4, 1.2]]
X = [[0.1, -0.3, 0.8], [1.2, 0.4, -0.5]]
M = [0.2, -0.1, 0.3]


def close(a, b, tol=1e-12):
    return all(abs(x - y) <= tol * max(1.0, abs(y)) for x, y in zip(a, b))


def test_normal_t_skew_normal_match_r():
    assert close(mvnormdens(X, mean=M, cov=S)["logpdf"], [-3.3469468664241, -3.887799338203])  # mvtnorm::dmvnorm
    assert close(mvtdens(X, loc=M, scale=S, df=4.5)["logpdf"], [-3.27529904307804, -4.04661280778315])  # mvtnorm::dmvt
    assert close(
        mvskewnorm(X, xi=M, omega=S, alpha=[2.0, -1.0, 0.5])["logpdf"], [-3.1667219621051, -3.49686708996871]
    )  # sn::dmsn
    r = mvnormdens(None, mean=M, cov=S, n=2, seed=3)["random"]
    assert close(r[0], [1.4765976693785927, -1.0853826822845107, 1.3957114540412003])  # Philox, identical in the R arm
    t = mvtdens(None, loc=M, scale=S, df=4.5, n=2, seed=3)["random"]
    assert close(t[1], [0.9053652107515255, -0.37820737917361147, 0.6757544893957833])


def test_bivariate_normal_cdf():
    got = [bvnormdist(0.7, -0.2, 0.1, 0.3, 1.2, 0.8, r)["cdf"] for r in (-0.6, 0.35, 0.95)]
    assert close(got, [0.104268667247688, 0.222352028950142, 0.265979615194317], 1e-13)  # mvtnorm::pmvnorm (TVPACK)
    for r in (-0.9, -0.3, 0.0, 0.5, 0.99):
        assert abs(bvnormdist(0.0, 0.0, rho=r)["cdf"] - (0.25 + math.asin(r) / (2 * math.pi))) < 1e-14
    from morie.fn._rrng_core import pnorm

    assert bvnormdist(0.3, 1.1, rho=1.0)["cdf"] == min(pnorm(0.3), pnorm(1.1))
    assert bvnormdist(0.3, 1.1, rho=-1.0)["cdf"] == max(0.0, pnorm(0.3) + pnorm(1.1) - 1)


def test_counts_wishart_lkj():
    assert close(
        [bvpois(3, 5, 1.2, 2.3, 0.7)["pmf"], bvpois(0, 2, 1.2, 2.3, 0.7)["pmf"]],
        [0.0223363190339444, 0.0396633006901635],
    )  # extraDistr
    assert close([multinomialdist([2, 3, 0, 1], probs=[0.1, 0.4, 0.2, 0.3])["pmf"]], [0.01152])  # dmultinom
    W = [[1.5, 0.2, 0.1], [0.2, 2.2, -0.3], [0.1, -0.3, 0.9]]
    assert close(
        [wishartdens(W, 5.5, S)["logpdf"], wishartdens(W, 5.5, S, inverse=True)["logpdf"]],
        [-11.5067626985679, -11.7980046163884],
    )  # MCMCpack
    # LKJ: the normaliser at eta = 1, d = 3 is the volume of the 3 x 3 elliptope, pi^2/2
    assert (
        abs(lkjcorr([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], 1.0)["log_normalizer"] - math.log(math.pi**2 / 2)) < 1e-14
    )
    # d = 2: r = 2 Beta(eta, eta) - 1, density (1 - r^2)^(eta - 1) / (2^(2 eta - 1) B(eta, eta))
    lb = 2 * math.lgamma(2.5) - math.lgamma(5.0)
    for r in (-0.9, -0.2, 0.0, 0.55, 0.97):
        exact = math.exp(1.5 * math.log(1 - r * r) - 4 * math.log(2) - lb)
        assert abs(lkjcorr([[1.0, r], [r, 1.0]], 2.5)["pdf"] - exact) < 1e-14 * max(1.0, exact)


def test_directional_densities():
    k = 4.2
    xs = [[1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [0.0, 1.0, 0.0]]
    got = vmfdens(xs, [0.0, 0.6, 0.8], k)["logpdf"]
    ref = [math.log(k / (4 * math.pi * math.sinh(k))) + k * (0.6 * x[1] + 0.8 * x[2]) for x in xs]
    assert close(got, ref, 1e-14)
    assert (
        abs(vmfdens([0, 0, 0, 0.6, 0.8], [0, 0, 0, 0, 1.0], 7.3)["logpdf"] + 1.01268132297945) < 1e-12
    )  # besselI(7.3, 1.5)
    # Kent density integrates to one over the sphere (product Gauss-Legendre in theta, phi)
    pts, w = [], []
    nodes = [
        (-0.9894009349916499, 0.0271524594117541),
        (-0.9445750230732326, 0.0622535239386479),
        (-0.8656312023878318, 0.0951585116824928),
        (-0.7554044083550030, 0.1246289712555339),
        (-0.6178762444026438, 0.1495959888165767),
        (-0.4580167776572274, 0.1691565193950025),
        (-0.2816035507792589, 0.1826034150449236),
        (-0.0950125098376374, 0.1894506104550685),
    ]
    nodes = nodes + [(-a, b) for a, b in reversed(nodes)]
    P = 12
    for i in range(P):
        for a, wa in nodes:
            th = math.pi * (i + (a + 1) / 2) / P
            for j in range(2 * P):
                for b, wb in nodes:
                    ph = 2 * math.pi * (j + (b + 1) / 2) / (2 * P)
                    pts.append([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
                    w.append(wa * wb / 4 * (math.pi / P) * (2 * math.pi / (2 * P)) * math.sin(th))
    dens = kentdens(pts, 5.0, 1.5)["pdf"]
    assert abs(sum(d * wt for d, wt in zip(dens, w)) - 1) < 1e-10
    assert abs(kentdens([0.0, 0.6, 0.8], 5.0, 1.5)["logpdf"] + 1.8637559207936132) < 1e-12  # R arm


def test_copula_densities_match_copula_package():
    got = [
        copuladens(0.3, 0.7, f, th)["density"]
        for f, th in (("gaussian", 0.5), ("t", 0.5), ("clayton", 2.0), ("gumbel", 1.7), ("frank", 3.0), ("joe", 2.2))
    ]
    ref = [
        0.877081937646637,
        0.831762144547868,
        0.629289451001217,
        0.78102664711419,
        0.769537139850275,
        0.772869856582706,
    ]
    assert close(got, ref, 1e-12)  # copula::dCopula
