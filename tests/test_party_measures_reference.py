"""Manifesto scales, nicheness and median voter (manifestoR)."""

from morie.fn._rng import random_uniform
from morie.fn.medvot import kim_fording_median
from morie.fn.mnfscl import manifesto_scales
from morie.fn.nichms import party_nicheness

CODES = [
    103,
    104,
    105,
    106,
    107,
    201,
    202,
    203,
    305,
    401,
    402,
    403,
    404,
    406,
    407,
    412,
    413,
    414,
    504,
    505,
    506,
    601,
    603,
    605,
    606,
    701,
]


def data():
    U = [float(u) for u in random_uniform(400, seed=41, stream=0)]
    docs = [{f"per{c}": round(8 * U[d * 26 + k], 3) for k, c in enumerate(CODES)} for d in range(5)]
    tot = [100 + int(300 * U[200 + d]) for d in range(5)]
    E = [[round(20 * U[300 + 5 * p + k], 3) for k in range(5)] for p in range(4)]
    for row in E:
        row[4] = 0.0
    w = [round(40 * U[380 + p], 3) for p in range(4)]
    return docs, tot, E, w


def close(a, b, tol):
    return all(abs(u - v) <= tol for u, v in zip(a, b))


def test_manifesto_scales_match_manifestor():
    docs, tot, _, _ = data()
    r = manifesto_scales(docs, total=tot)
    # rile(D); logit_rile(D); scale_ratio_1/2(D, pos = rile_r(), neg = rile_l())
    assert close(r.value, [-5.726, 20.709, -8.523, -13.139, -2.7], 1e-9)
    assert close(
        r.extra["logit"],
        [-0.1403552542434652, 0.4777129478813487, -0.1729726487187145, -0.2382435461571807, -0.0493194763887821],
        1e-12,
    )
    assert close(
        r.extra["ratio_1"],
        [-0.0708365291832645, 0.2356857523302263, -0.0865042069686482, -0.1190612115445607, -0.0247447623586341],
        1e-12,
    )
    assert close(
        r.extra["ratio_2"],
        [0.867698706099815, 1.616724738675958, 0.840765997197571, 0.787212334202471, 0.951705510937807],
        1e-12,
    )


def test_nicheness_matches_meyer_miller_single_election():
    _, _, E, w = data()
    ref = [-0.194917642272221, 2.226027822681774, -1.327398180565975, -1.669903719483974]
    assert close(party_nicheness(E, w).value, ref, 1e-12)
    ref_b = [-0.2051248456044528, 0.2248952807480336, 0.0491882523947528, -0.1108839855779615]
    assert close(party_nicheness(E, w, transform="bischof").value, ref_b, 1e-12)
    assert party_nicheness([[10, 0], [4, 6], [2, 8]], normalize=False).value == [7.0, 2.0, 5.0]


def test_kim_fording_median():
    assert kim_fording_median([-20, 5, 30], [30, 25, 45]).value == 12.5
    # manifestoR:::median_voter_single(c(-20, 5, 30, 5), c(30, 25, 45, 10), adjusted = TRUE)
    assert abs(kim_fording_median([-20, 5, 30, 5], [30, 25, 45, 10], adjusted=True).value - 10.3571428571429) < 1e-12
