"""vgmods against gstat 2.1 (variogramLine, variogram, fit.variogram) and geoR 1.9 (cov.spatial, likfit, loglik.GRF).

Reference values printed by R on the data below; the WLS fits use the
Philox field FIELD40 (regenerated here and in R), where gstat converges.
Optimiser outputs agree to the optimisers' tolerance: the weighted SSE and
the log-likelihood exactly, parameters on flat objectives to 1e-3.
"""

import math

import pytest

from morie.fn._rng import random_uniform
from morie.fn.vgmods import (
    fit_variogram,
    likfit,
    sample_variogram,
    select_variogram_model,
    variogram_loglik,
    variogram_map,
    vgm_correlogram,
    vgm_covariance,
    vgm_semivariance,
    vgm_spectral_density,
)
from morie.fn.zschl import chol_sim

X = [
    2.7840530000000001,
    0.60747600000000002,
    4.4060319999999997,
    1.080843,
    3.5662419999999999,
    3.2673230000000002,
    1.3339650000000001,
    2.8159299999999998,
    1.803682,
    5.2816789999999996,
    3.1841590000000002,
    1.6519159999999999,
    4.8301879999999997,
    4.9097689999999998,
    5.6265460000000003,
    1.3132550000000001,
    0.261957,
    4.8379960000000004,
    5.7355650000000002,
    2.2019669999999998,
    5.6833,
    3.9661550000000001,
    2.2457310000000001,
    1.3691409999999999,
    0.212561,
    3.3552550000000001,
    5.3880080000000001,
    4.2531980000000003,
    4.8446290000000003,
    5.6811860000000003,
]
Y = [
    1.371005,
    3.0127060000000001,
    1.0539510000000001,
    2.6276250000000001,
    0.45469100000000001,
    0.64088599999999996,
    2.2267999999999999,
    5.409205,
    3.6378180000000002,
    4.7837880000000004,
    4.3290430000000004,
    5.491981,
    1.9798420000000001,
    4.6575110000000004,
    3.121146,
    1.1787879999999999,
    2.1762069999999998,
    0.76300400000000002,
    2.2702629999999999,
    4.1796800000000003,
    1.052276,
    0.48547099999999999,
    2.5416120000000002,
    5.9711829999999999,
    3.7942390000000001,
    5.4595710000000004,
    5.6701079999999999,
    3.1404420000000002,
    5.1373490000000004,
    0.25842399999999999,
]
Z = [
    2.43588,
    3.1441170000000001,
    0.97847499999999998,
    1.8258890000000001,
    1.1166529999999999,
    1.8431219999999999,
    0.76476100000000002,
    0.95067100000000004,
    1.8209679999999999,
    -1.005153,
    2.1863579999999998,
    1.8699300000000001,
    1.7243809999999999,
    0.82542800000000005,
    2.5325850000000001,
    1.503063,
    0.72356600000000004,
    0.24885699999999999,
    1.8725879999999999,
    2.508435,
    1.4701880000000001,
    0.82920400000000005,
    2.6913999999999998,
    1.04914,
    0.732402,
    2.147824,
    0.301792,
    2.4322059999999999,
    0.77937699999999999,
    -0.020441999999999998,
]
LINES = {
    "Exp": [0.18107963064742486, 0.4710834028916947, 0.74436058846665532, 0.94570866905578377],
    "Sph": [0.29030624999999999, 0.81826874999999999, 1.2583187499999999, 1.3],
    "Gau": [0.028923391648662755, 0.23830757262245592, 0.6688020362996967, 1.060124618809114],
    "Exc": [0.073371297144897471, 0.33873476823162685, 0.70625005389416151, 1.0047296164932105],
    "Mat": [0.013241575244538317, 0.098070934192957399, 0.27206708866331208, 0.48512993882830241],
    "Ste": [0.068948001606647283, 0.39234254494754084, 0.8004770691164681, 1.07476798101626],
    "Cir": [0.24734748696030578, 0.71888164376194763, 1.211408327957419, 1.3],
    "Lin": [0.19500000000000001, 0.58500000000000008, 1.105, 1.3],
    "Bes": [0.036887738650152935, 0.19345984943975747, 0.4329329305364415, 0.67039473238198399],
    "Pen": [0.36017764453125001, 0.95779262109374996, 1.2902282148437501, 1.3],
    "Per": [0.53587917201985014, 2.5363734711837744, 0.53587917201941526, 1.7017220926880967],
    "Wav": [0.04758290939465544, 0.39175943941368208, 1.0789852193050289, 1.5575181074002753],
    "Hol": [0.0048695185621400738, 0.043432901456446037, 0.15098290978543458, 0.33644181458280703],
    "Log": [1.0827818598156351, 1.3841239580901568, 1.7008326655452326, 1.9838731945435639],
    "Pow": [0.21361179742701475, 1.1099594587191013, 2.8814874630995706, 5.4500862378498205],
}
GEOR_COV = {
    "Cub": [1.1332967752539063, 0.41347985529296899, 0.0046253641992194394, 0],
    "Cau": [1.2573273923997916, 0.98586108667579053, 0.57504854170525244, 0.29465575803144983],
}
SV = {
    "np": [1, 5, 5, 7, 12, 12, 10, 12, 19, 9, 20, 11, 14, 16],
    "dist": [
        0.35216636208757901,
        0.4545933510616586,
        0.58858965578493594,
        0.79979793321932069,
        0.96091726649854181,
        1.1597453094745738,
        1.3161974371000806,
        1.4949439271759781,
        1.6871242190083471,
        1.8824206466224085,
        2.0368031829862869,
        2.1617517662733334,
        2.4082876101599107,
        2.5493525579847947,
    ],
    "gamma": [
        0.2638786039805,
        0.50941094780710006,
        0.75017547864820011,
        0.72169314993485723,
        0.65172741188120853,
        0.61018763783008334,
        0.32332090878270003,
        0.51823885232404165,
        0.89751288376855265,
        1.0343987988456109,
        0.95867892689754997,
        1.1927579414619089,
        0.70503990847910702,
        1.184420802805094,
    ],
}
SV_CR = {
    "np": [1, 5, 5, 7, 12, 12, 10, 12, 19, 9, 20, 11, 14, 16],
    "dist": [
        0.35216636208757901,
        0.4545933510616586,
        0.58858965578493594,
        0.79979793321932069,
        0.96091726649854181,
        1.1597453094745738,
        1.3161974371000806,
        1.4949439271759781,
        1.6871242190083471,
        1.8824206466224085,
        2.0368031829862869,
        2.1617517662733334,
        2.4082876101599107,
        2.5493525579847947,
    ],
    "gamma": [
        0.27747487274500526,
        0.36408139613466384,
        1.1471380539372813,
        0.67918944341021359,
        0.60418188217695545,
        0.59432418595880632,
        0.26067543093841594,
        0.68502776347437089,
        0.85506766641259069,
        0.5790287171162074,
        1.0710017945112678,
        1.3206881743459884,
        0.58121335853393874,
        1.5775246356720714,
    ],
}
SV_COV = {
    "np": [1, 5, 5, 7, 12, 12, 10, 12, 19, 9, 20, 11, 14, 16, 30],
    "dist": [
        0.35216636208757901,
        0.4545933510616586,
        0.58858965578493594,
        0.79979793321932069,
        0.96091726649854181,
        1.1597453094745738,
        1.3161974371000806,
        1.4949439271759781,
        1.6871242190083471,
        1.8824206466224085,
        2.0368031829862869,
        2.1617517662733334,
        2.4082876101599107,
        2.5493525579847947,
        0,
    ],
    "gamma": [
        -0.12697863536624998,
        0.43596059229694967,
        0.43826794804565006,
        0.057191457833535728,
        0.29707973822287487,
        0.017773653157000052,
        0.38540783848485016,
        -0.036099550461374981,
        0.027013650716329055,
        -0.24475829423502779,
        -0.078843082388199828,
        -0.3208495042770228,
        -0.058555697826499879,
        -0.20658311684978126,
        0.8306271608808502,
    ],
}
SV_DIR = {
    "np": [2, 2, 5, 4, 8, 6, 4, 10, 5, 9, 4, 8, 8, 1, 3, 3, 2, 8, 4, 4, 8, 9, 4, 11, 7, 6, 8],
    "dist": [
        0.47914889563126273,
        0.61445282803308809,
        0.82433979797847901,
        0.96609062419731861,
        1.1621770158578189,
        1.308822490637825,
        1.457923163771861,
        1.6757091847586678,
        1.9122400827375206,
        2.026382664141392,
        2.1708133646791676,
        2.4149551929994195,
        2.5206657355205158,
        0.35216636208757901,
        0.43822298801525578,
        0.57134754095283447,
        0.73844327132142473,
        0.95833058764915346,
        1.1548818967080838,
        1.3272598567934637,
        1.5134543088780372,
        1.6998075903968801,
        1.8451463514785185,
        2.0453290620412004,
        2.1565737100414286,
        2.3993974997072334,
        2.5780393804490735,
    ],
    "gamma": [
        0.28202833174625003,
        0.28657677504725004,
        0.88475869561390008,
        1.08357974758325,
        0.76881534031787502,
        0.30923061531891666,
        0.35289903841487497,
        1.1542867783191002,
        1.5329157620650999,
        0.87565806833383319,
        1.015960353201,
        0.67769408718087498,
        1.3603658643901877,
        0.2638786039805,
        0.66099935851433356,
        1.0592412810488334,
        0.31402928573724992,
        0.4358012440301875,
        0.29293223285449993,
        0.34445634897837507,
        0.60090875927862497,
        0.61220855649016659,
        0.41125259482124987,
        1.0266050839042273,
        1.2937851347538571,
        0.74150100354341664,
        1.0084757412199998,
    ],
}
FITS_NOISY = {
    1: [0.44914236988925005, 13.822153327442262, 51.67297402046956, 6.4869810701308248],
    6: [0.393367537831942, 27.601657575764996, 98.431991718824037, 0.5757613219063944],
    7: [0.43104943474568763, 23.253073468655359, 92.041941990578295, 3.204179128120241],
}
LIKFIT = {
    "ML": [0.73080314029876281, 0.6870469047489759, 0.10894348926670085, -38.320863027927587, 1.4107706060845486],
    "REML": [0.73315582516562106, 1.340579326040223, 0.25847474993729436, -36.949937605553636, 1.2699724057521582],
}
LOGLIK_AT = [-38.654044073037724, -37.02726453844916]
FIT40 = {
    1: [0.36583408124640204, 0.66634407798827544, 5.2978571490734554, 17.155097590681635],
    6: [0.43151393240078056, 0.6407098504105383, 6.1565123376801552, 0.7322266890843403],
    7: [0.43981722167784892, 1.3041288051335065, 14.007268533820726, 3.4065717992905231],
}

P = list(zip(X, Y))
H = [0.3, 0.9, 1.7, 2.6]


def _field40():
    u = [float(v) for v in random_uniform(120, seed=9, stream=0)]
    P40 = [(10 * u[i], 10 * u[40 + i]) for i in range(40)]
    z = chol_sim(P40, [{"model": "Nug", "psill": 0.2}, {"model": "Sph", "psill": 1.0, "range": 4.0}], seed=2)
    return z.simulations[0], P40


@pytest.mark.parametrize("m", sorted(LINES))
def test_gstat_variogram_lines(m):
    c = {"model": m, "psill": 1.3, "range": 1.5 if m == "Pow" else 2.0}
    if m in ("Exc", "Mat", "Ste"):
        c["kappa"] = 1.5
    # gstat's periodic model carries a truncated pi (about 3e-13 off)
    assert vgm_semivariance(H, c) == pytest.approx(LINES[m], abs=1e-11 if m == "Per" else 1e-13)


def test_geor_cubic_and_cauchy_covariances():
    assert vgm_covariance(H, {"model": "Cub", "psill": 1.3, "range": 2.0}) == pytest.approx(GEOR_COV["Cub"], abs=1e-14)
    cau = vgm_covariance(H, {"model": "Cau", "psill": 1.3, "range": 2.0, "kappa": 1.5})
    assert cau == pytest.approx(GEOR_COV["Cau"], abs=1e-14)


def test_jbessel_damped_cosine_and_identities():
    # J_{1/2}: Gamma(3/2)(2/x)^{1/2} J_{1/2}(x) = sin(x)/x, the hole effect
    jb = vgm_semivariance(H, {"model": "JBes", "psill": 1.0, "range": 2.0, "kappa": 0.5})
    assert jb == pytest.approx(vgm_semivariance(H, {"model": "Hol", "psill": 1.0, "range": 2.0}), abs=1e-14)
    d = vgm_semivariance(1.3, {"model": "Dmp", "psill": 2.0, "range": 1.0, "period": 0.5})
    assert d == pytest.approx(2.0 * (1.0 - math.exp(-1.3) * math.cos(2.6)), abs=1e-15)
    assert vgm_semivariance(0.7, {"model": "Cos", "psill": 1.0, "range": 0.5}) == pytest.approx(1.0 - math.cos(1.4))
    m = [{"model": "Nug", "psill": 0.2}, {"model": "Sph", "psill": 1.0, "range": 2.0}]
    assert vgm_semivariance(0.0, m) == 0.0
    assert vgm_covariance(0.5, m) == pytest.approx(1.2 - vgm_semivariance(0.5, m), abs=1e-15)
    assert vgm_correlogram(0.5, m) == pytest.approx(1.0 - vgm_semivariance(0.5, m) / 1.2, abs=1e-15)
    with pytest.raises(ValueError):
        vgm_covariance(1.0, {"model": "Pow", "range": 1.0})


def test_spectral_density_inverts_to_the_covariance():
    # 1-D: C(h) = 2 int_0^inf f(w) cos(w h) dw; midpoint rule on [0, 400]
    for m in ({"model": "Gau", "psill": 1.0, "range": 1.3}, {"model": "Mat", "psill": 1.0, "range": 0.8, "kappa": 1.5}):
        for h in (0.0, 0.7):
            dw = 0.002
            s = (
                2.0
                * dw
                * sum(
                    vgm_spectral_density((k + 0.5) * dw, m, d=1) * math.cos((k + 0.5) * dw * h) for k in range(200000)
                )
            )
            assert s == pytest.approx(vgm_covariance(h, m), abs=1e-6)


def _chk_sv(out, ref):
    assert out.np == ref["np"]
    assert out.dist == pytest.approx(ref["dist"], abs=1e-13)
    assert out.gamma == pytest.approx(ref["gamma"], abs=1e-13)


def test_sample_variograms_equal_gstat():
    _chk_sv(sample_variogram(Z, P), SV)
    _chk_sv(sample_variogram(Z, P, estimator="cressie"), SV_CR)
    _chk_sv(sample_variogram(Z, P, covariogram=True), SV_COV)
    _chk_sv(sample_variogram(Z, P, alpha=[0, 90]), SV_DIR)


def test_other_estimators_by_hand():
    z = [1.0, 2.0, 4.0, 3.0]
    q = [(0, 0), (1, 0), (0, 1), (1, 1)]
    b = [0, 1.2]
    dz = [-1.0, -3.0, -1.0, 1.0]  # pairs (0,1), (0,2), (1,3), (2,3)
    zi, zj = [1.0, 1.0, 2.0, 4.0], [2.0, 4.0, 3.0, 3.0]
    assert sample_variogram(z, q, boundaries=b, estimator="mad").gamma[0] == pytest.approx(1.099 * 1.0, abs=1e-15)
    pr = sum((d / ((a + c) / 2)) ** 2 for d, a, c in zip(dz, zi, zj)) / 8
    assert sample_variogram(z, q, boundaries=b, estimator="pairwise_relative").gamma[0] == pytest.approx(pr, abs=1e-15)
    rel = (12.0 / 8) / (sum(zi + zj) / 8) ** 2
    assert sample_variogram(z, q, boundaries=b, estimator="relative").gamma[0] == pytest.approx(rel, abs=1e-15)
    ind = sample_variogram(z, q, boundaries=b, threshold=2.5)
    assert ind.gamma[0] == pytest.approx((0 + 1 + 1 + 0) / 8, abs=1e-15)
    px = sample_variogram(z, q, boundaries=b, z2=[0.0, 1.0, 1.0, 2.0])
    want = [
        (z[i] - [0.0, 1.0, 1.0, 2.0][j]) ** 2
        for i in range(4)
        for j in range(4)
        if i != j and math.dist(q[i], q[j]) <= 1.2
    ]
    assert px.gamma[0] == pytest.approx(sum(want) / (2 * len(want)), abs=1e-15)
    assert len(sample_variogram(Z, P, cloud=True).gamma) == 30 * 29 // 2
    r = variogram_map([1.0, 2.0, 4.0], [(0, 0), (1, 0), (0, 1)], cutoff=1.0, width=1.0)
    cells = dict(zip(zip(r.dx, r.dy), r.gamma))
    assert cells[(1.0, 0.0)] == cells[(-1.0, 0.0)] == 0.5
    assert cells[(1.0, -1.0)] == cells[(-1.0, 1.0)] == 2.0


@pytest.mark.parametrize("method", [1, 6, 7])
def test_wls_fits_reach_the_gstat_optimum(method):
    z, P40 = _field40()
    sv = sample_variogram(z, P40)
    f = fit_variogram(sv, [{"model": "Nug", "psill": 0.1}, {"model": "Sph", "psill": 1.0, "range": 3.0}], method=method)
    g = FIT40[method]
    assert f.sse <= g[3] * (1 + 1e-9)
    assert f.sse == pytest.approx(g[3], rel=1e-9)
    assert [f.model[0]["psill"], f.model[1]["psill"], f.model[1]["range"]] == pytest.approx(g[:3], rel=1e-3)


def test_cressie_criterion_fit_is_no_worse_than_its_start():
    z, P40 = _field40()
    sv = sample_variogram(z, P40)
    m0 = [{"model": "Nug", "psill": 0.1}, {"model": "Sph", "psill": 1.0, "range": 3.0}]
    f2 = fit_variogram(sv, m0, method=2)
    f7 = fit_variogram(sv, m0, method=7)
    g7 = vgm_semivariance(sv.dist, f7.model)
    c7 = sum(n * (o - e) ** 2 / e**2 for n, o, e in zip(sv.np, sv.gamma, g7))
    assert f2.sse <= c7


def test_loglik_equals_geor():
    m = {"model": "Exp", "psill": 0.9, "range": 1.2}
    assert variogram_loglik(Z, P, m, nugget=0.2) == pytest.approx(LOGLIK_AT[0], abs=1e-11)
    assert variogram_loglik(Z, P, m, nugget=0.2, method="REML") == pytest.approx(LOGLIK_AT[1], abs=1e-11)


@pytest.mark.parametrize("meth", ["ML", "REML"])
def test_likfit_equals_geor(meth):
    f = likfit(Z, P, {"model": "Exp", "range": 1.5}, method=meth, nugget=0.3)
    g = LIKFIT[meth]
    assert f.loglik == pytest.approx(g[3], abs=1e-6)
    assert f.loglik >= g[3] - 1e-9
    assert [f.psill, f.range, f.nugget] == pytest.approx(g[:3], rel=2e-3)
    assert f.beta[0] == pytest.approx(g[4], abs=1e-3)
    assert pytest.approx(-2 * f.loglik + 2 * 4, abs=1e-12) == f.AIC


def test_model_selection_picks_the_smallest_aic():
    s = select_variogram_model(Z, P, [{"model": "Exp", "range": 1.5}, {"model": "Gau", "range": 1.5}])
    assert s.best == min(range(2), key=lambda i: s.AIC[i])
    assert s.AIC[0] == pytest.approx(-2 * LIKFIT["ML"][3] + 8, abs=1e-5)
