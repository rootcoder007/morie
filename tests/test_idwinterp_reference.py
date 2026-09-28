"""idwinterp against gstat 2.1 (idw with idp/nmax/maxdist/block, krige.cv) plus hand-derived Shepard weights."""

import pytest

from morie.fn._rng import random_uniform
from morie.fn.idwinterp import idw_cv, idw_power_search, idw_predict, shepard_predict

GS = {
    "p2": [7.0091797675761915, 7.8229918072354474, 7.8957430192512552, 6.9981],
    "nmax": [6.9756265222930489, 7.9612663795098397, 8.0633311438541551, 6.9981],
    "maxd": [7.1687919495442864, 8.0988615197356069, 7.9071999999999996, 6.9981],
    "block": [6.9368733522395054, 7.8319993286989948, 7.8974777146782733],
    "cv": [
        7.9136828165128508,
        6.621255599918106,
        8.219070266439596,
        6.6665073197627178,
        8.3774119864984353,
        6.2713652102526343,
        7.4402581630917295,
        5.6497523858089789,
        6.5825765556629152,
        6.5457978949584161,
        6.9349407542383492,
        8.1693721767373226,
        7.8405979550963911,
        7.5198174731876826,
        7.8517657557741973,
        6.3074185729471361,
        7.4941050190008527,
        7.1785635248446766,
        7.922234895005098,
        8.4895666539453654,
        8.1301782421730362,
        7.9320928542625717,
        6.871205829668301,
        8.3870882324289067,
        7.0756810798647471,
        8.0868002312700451,
        7.4526425584550511,
        7.6749929412652396,
        6.6764512615819349,
        5.7934145447560432,
    ],
}

_u = [float(v) for v in random_uniform(200, seed=91, stream=0)]
P = [(10 * _u[i], 10 * _u[30 + i]) for i in range(30)]
Z = [round(5 + 2 * _u[60 + i] + 0.3 * P[i][0], 4) for i in range(30)]
Q = [(1.5, 2.5), (5.0, 5.0), (8.2, 1.1), P[3]]
BLK = [(dx, dy) for dx in (-0.25, 0.25) for dy in (-0.25, 0.25)]


def _same(a, b):
    for x, y in zip(a, b):
        assert (x != x and y != y) or x == pytest.approx(y, abs=1e-12)


def test_idw_equals_gstat():
    _same(idw_predict(Z, P, Q).prediction, GS["p2"])
    _same(idw_predict(Z, P, Q, nmax=5).prediction, GS["nmax"])
    _same(idw_predict(Z, P, Q, maxdist=2.5, power=3.5).prediction, GS["maxd"])
    _same(idw_predict(Z, P, Q[:3], block=BLK).prediction, GS["block"])
    _same(idw_cv(Z, P, nmax=8).prediction, GS["cv"])
    assert idw_predict(Z, P, [P[3]]).prediction[0] == Z[3]


def test_power_search_anisotropy_and_shepard_by_hand():
    s = idw_power_search(Z, P, [1.0, 2.0, 3.0])
    assert s.rmse == pytest.approx([idw_cv(Z, P, power=p).rmse for p in (1.0, 2.0, 3.0)])
    assert s.best == [1.0, 2.0, 3.0][min(range(3), key=lambda i: s.rmse[i])]
    # anisotropy: ratio 0.5 along x doubles cross-axis distances
    a = idw_predict([1.0, 3.0], [(0, 1), (1, 0)], [(0, 0)], angle=0.0, ratio=0.5).prediction[0]
    w1, w2 = 1 / 2.0**2, 1 / 1.0**2
    assert a == pytest.approx((w1 * 1 + w2 * 3) / (w1 + w2), abs=1e-15)
    sh = shepard_predict([1.0, 3.0], [(0, 0), (2, 0)], [(0.5, 0)], radius=3.0).prediction[0]
    w = [((3 - 0.5) / (3 * 0.5)) ** 2, ((3 - 1.5) / (3 * 1.5)) ** 2]
    assert sh == pytest.approx((w[0] + 3 * w[1]) / sum(w), abs=1e-15)
    with pytest.raises(ValueError):
        shepard_predict([1.0], [(0, 0)], [(1, 1)])
