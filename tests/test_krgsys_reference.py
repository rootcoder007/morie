"""krgsys kriging core against gstat 2.1 (krige, predict BLUE, krige.cv).

Reference values: gstat::krige on the data below
(tests/cross/test-morie_vs_gstat_krige.R repeats the comparison live in R).
gstat keeps its Gauss-Legendre block weights in single precision, so the
default (Gauss) block case agrees to 5e-8; everything else to 1e-12.
"""

import math

import pytest

from morie.fn.krgsys import (
    block_discretize,
    factorial_krige,
    krige,
    krige_cv,
    krige_lognormal,
    kriging_covariance,
    kriging_exceedance,
    kriging_quantile,
    kriging_system,
)

P = [(0, 0), (1.2, 0.3), (0.4, 1.7), (2.1, 1.1), (1.5, 2.4), (2.8, 0.2), (0.9, 2.9), (2.6, 2.7)]
Z = [1.0, 1.8, 0.7, 2.4, 1.3, 2.9, 0.4, 2.2]
Q = [(0.8, 0.9), (2.0, 2.0)]
M = {"model": "Exp", "psill": 1.0, "range": 1.5, "nugget": 0.1}

GSTAT = {
    "OK": (1.3700319610959943, 1.8336566372755998, 0.58712741628861664, 0.55607174262983783),
    "SK": (1.3657321296988003, 1.8290680140023887, 0.58652345365719227, 0.55538392702844674),
    "UK": (1.3046153589941756, 1.9094289827992641, 0.58824215244443556, 0.55773238768700539),
    "BLUE": (1.2393540097711828, 1.8049759913994421, 0.45043853952068652, 0.46898741381487008),
    "MAT": (1.3111612056617858, 1.9493259459676757, 0.29587112535546217, 0.25321891102650212),
    "GAU": (1.334190251892128, 1.9514699058625091, 0.72461943747407631, 0.59736695997236566),
    "BLKG": (1.3676275906907347, 1.8350997220702985, 0.33303983608730015, 0.30408642636801864),
    "BLKR": (1.3677686534515356, 1.8350214291437226, 0.33658254357302375, 0.30749642681615502),
    "NMAX": (1.3817297908732462, 1.8451105843329656, 0.59058574824526833, 0.56056782493370072),
    "D3": (1.4154591913838646, 1.7930318223272221, 0.77554763841480912, 0.63410323683111414),
}
LOO_Z = (
    -0.57682919341083627,
    0.13032673208744461,
    -0.60586342354595335,
    0.46892436160634643,
    0.024918872421163512,
    1.0488296060048288,
    -1.0854814655898963,
    0.66070009595194623,
)
KF_PRED = (
    1.6450348252483711,
    1.664903775112877,
    1.2212659345795875,
    2.0158706183529813,
    1.0930697215930603,
    1.8926061159407896,
    1.3921571994888182,
    1.4806327957246705,
)


def _chk(r, key, tol=1e-12):
    g = GSTAT[key]
    assert r.prediction == pytest.approx(list(g[:2]), abs=tol)
    assert r.variance == pytest.approx(list(g[2:]), abs=tol)


def test_ordinary_simple_universal_blue():
    _chk(krige(Z, P, Q, M), "OK")
    _chk(krige(Z, P, Q, M, beta=1.5), "SK")
    X = [[1.0, x, y] for x, y in P]
    X0 = [[1.0, x, y] for x, y in Q]
    _chk(krige(Z, P, Q, M, X=X, X0=X0), "UK")
    _chk(krige(Z, P, Q, M, X=X, X0=X0, blue=True), "BLUE")


def test_matern_gaussian_models():
    _chk(krige(Z, P, Q, {"model": "Mat", "psill": 0.8, "range": 0.7, "kappa": 1.5, "nugget": 0.05}), "MAT")
    _chk(krige(Z, P, Q, [{"model": "Nug", "psill": 0.2}, {"model": "Gau", "psill": 1.2, "range": 1.1}]), "GAU")


def test_block_kriging():
    _chk(krige(Z, P, Q, M, block=block_discretize((0.4, 0.6))), "BLKG", tol=5e-8)
    _chk(krige(Z, P, Q, M, block=block_discretize((0.4, 0.6), 4, "regular")["offsets"]), "BLKR")


def test_local_neighbourhood_and_3d():
    _chk(krige(Z, P, Q, M, nmax=4), "NMAX")
    P3 = [(x, y, 0.3 * i) for i, (x, y) in enumerate(P)]
    _chk(krige(Z, P3, [(x, y, 1.0) for x, y in Q], {"model": "Sph", "psill": 1.0, "range": 2.5, "nugget": 0.1}), "D3")
    assert math.isnan(krige(Z, P, Q, M, maxdist=0.1).prediction[0])


def test_cross_validation():
    r = krige_cv(Z, P, M)
    assert r.zscore == pytest.approx(list(LOO_Z), abs=1e-12)
    assert r.rmse == pytest.approx(math.sqrt(sum(v * v for v in r.residual) / 8), abs=1e-15)
    assert r.mae == pytest.approx(sum(abs(v) for v in r.residual) / 8, abs=1e-15)
    k = krige_cv(Z, P, M, folds=[1, 2, 3, 1, 2, 3, 1, 2])
    assert k.prediction == pytest.approx(list(KF_PRED), abs=1e-12)


def test_ordinary_kriging_is_an_exact_interpolator_and_system_agrees():
    r = krige(Z, P, P[:3], M)
    assert r.prediction == pytest.approx(Z[:3], abs=1e-12)
    assert r.variance == pytest.approx([0.0] * 3, abs=1e-12)
    s = kriging_system(P, Q[0], M)
    k = krige(Z, P, Q[:1], M)
    assert s.weights == pytest.approx(k.weights[0], abs=1e-12)
    assert s.lagrange == pytest.approx(k.lagrange[0], abs=1e-12)
    assert sum(s.weights) == pytest.approx(1.0, abs=1e-12)


def test_gls_beta_and_trend_residuals():
    r = krige(Z, P, Q, M)
    # the OK mean is the GLS mean 1' C^-1 z / 1' C^-1 1; the residuals remove it
    assert r.trend_residuals == pytest.approx([v - r.beta[0] for v in Z], abs=1e-15)


def test_lognormal_back_transform():
    m = {"model": "Exp", "psill": 0.3, "range": 2.0}
    r = krige_lognormal(Z, P, Q, m)
    k = krige([math.log(v) for v in Z], P, Q, m)
    want = [math.exp(y + s / 2 + mu[0]) for y, s, mu in zip(k.prediction, k.variance, k.lagrange)]
    assert r.prediction == pytest.approx(want, abs=1e-14)
    s = krige_lognormal(Z, P, Q, m, beta=0.3)
    ks = krige([math.log(v) for v in Z], P, Q, m, beta=0.3)
    assert s.prediction == pytest.approx([math.exp(y + v / 2) for y, v in zip(ks.prediction, ks.variance)], abs=1e-14)
    assert round(krige_lognormal([1.0, 3.0, 2.0], [(0, 0), (2, 0), (0, 2)], [(1, 1)], m).prediction[0], 6) == 2.027853


def test_factorial_kriging_components_sum_to_the_filtered_prediction():
    m = [{"model": "Exp", "psill": 1.0, "range": 1.0}, {"model": "Sph", "psill": 0.5, "range": 4.0}]
    a = factorial_krige(Z, P, Q, m, 0).prediction
    b = factorial_krige(Z, P, Q, m, 1).prediction
    ok = krige(Z, P, Q, m)
    # the OK prediction is the GLS mean plus the two component estimates
    assert [x + y + ok.beta[0] for x, y in zip(a, b)] == pytest.approx(ok.prediction, abs=1e-12)
    r = factorial_krige([1.0, 3.0, 2.0, 2.5], [(0, 0), (2, 0), (0, 2), (2, 2)], [(0.5, 0.2)], m, 1)
    assert round(r.prediction[0], 6) == -0.175133


def test_quantile_exceedance_covariance_and_discretisation():
    assert kriging_quantile([1.0], [0.25], 0.975)[0] == pytest.approx(1.0 + 1.959963984540054 * 0.5, abs=1e-12)
    assert kriging_exceedance([1.0], [0.25], 1.5)[0] == pytest.approx(0.15865525393145705, abs=1e-14)
    assert kriging_covariance(0.0, M) == pytest.approx(1.1, abs=1e-15)
    assert kriging_covariance(1.5, M) == pytest.approx(math.exp(-1.0), abs=1e-15)
    g = block_discretize((0.4, 0.6))
    assert sum(g["weights"]) == pytest.approx(1.0, abs=1e-14)
    assert g["offsets"][0][0] == pytest.approx(0.2 * 0.8611363115940526, abs=1e-15)


def test_validation():
    with pytest.raises(ValueError):
        krige(Z, P, Q, M, X=[[1.0]] * 8)
    with pytest.raises(ValueError):
        krige_lognormal([0.0, 1.0], [(0, 0), (1, 0)], [(0.5, 0)], M)
    with pytest.raises(ValueError):
        factorial_krige(Z, P, Q, M, 3)
    with pytest.raises(ValueError):
        kriging_quantile([1.0], [1.0], 1.0)
