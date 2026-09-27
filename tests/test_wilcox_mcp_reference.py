"""Studentized range (port of R's ptukey/qtukey), Tukey-Kramer (12.4), Scheffe (12.7-12.9), Johansen (10.3-10.4)."""

import math

from morie.fn._rrng_core import qchisq, qf
from morie.fn.johanq import johanq
from morie.fn.ptukey import ptukey
from morie.fn.qtukey import qtukey
from morie.fn.scheffeci import scheffeci
from morie.fn.trimse import _tmean
from morie.fn.tukeykramer import tukeykramer
from morie.fn.yuen import _yuen_d

G1 = [2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 6.1, 3.9, 4.2, 2.2, 5.0, 40.0]
G2 = [3.3, 4.1, 2.7, 6.8, 5.9, 4.4, 3.6, 7.2, 4.8, 5.3, 3.1, 6.6, 4.0, 9.5, 5.5]
G3 = [5.1, 6.3, 4.8, 7.7, 6.9, 5.5, 6.1, 8.4, 5.9, 7.0, 4.6]
G4 = [1.2, 2.8, 3.3, 2.1, 4.4, 3.9, 2.5, 3.0, 1.8, 2.6]


def test_studentized_range_matches_r():
    ref = {
        (3.5, 3, 20): 0.94410815091794065,
        (2.0, 5, 10): 0.36691773747042844,
        (4.2, 4, 60): 0.97833824714964623,
        (1.0, 2, 5): 0.48891591956964403,
        (5.5, 8, 200): 0.9966221166669148,
        (3.0, 3, 1000): 0.91394580847797324,
        (3.3, 6, 30000): 0.81939837333522825,
        (0.3, 2, 2): 0.14834435149975123,
    }
    for a, p in ref.items():
        assert abs(ptukey(*a)["p"] - p) < 1e-13
    # two means: the range is |Z1 - Z2| / s, so P(range <= q) = P(|T_df| <= q / sqrt 2)
    from morie.fn._rrng_core import pt

    assert abs(ptukey(2.5, 2, 12)["p"] - (1 - 2 * pt(-2.5 / math.sqrt(2), 12))) < 1e-9
    for (p, k, df), q in {
        (0.95, 3, 20): 3.577934581525569,
        (0.99, 5, 10): 6.136093312662168,
        (0.9, 4, 60): 3.3119043396336911,
        (0.95, 10, 300): 4.5081268179442437,
    }.items():
        r = qtukey(p, k, df)["q"]
        assert abs(r - q) < 1e-12 and abs(ptukey(r, k, df)["p"] - p) < 1e-6


def test_tukey_kramer_matches_tukeyhsd():
    r = tukeykramer([G1, G2, G3])
    m = [sum(g) / len(g) for g in (G1, G2, G3)]
    mswg = sum(sum((v - mm) ** 2 for v in g) for g, mm in zip((G1, G2, G3), m)) / 36
    assert r["df"] == 36 and abs(r["mswg"] - mswg) < 1e-12
    c = r["comparisons"][0]
    half = r["q"] * math.sqrt(mswg / 2 * (1 / 13 + 1 / 15))
    assert abs(c["upper"] - (c["diff"] + half)) < 1e-12
    # R: TukeyHSD(aov(v ~ f)) reports level 2 minus level 1
    assert abs(c["diff"] - 1.410769230769230) < 1e-12 and abs(c["upper"] - 6.96982568114729) < 1e-9
    assert abs(c["p_adj"] - 0.809996319496019) < 1e-9
    assert abs(r["comparisons"][2]["lower"] + 6.91259085892739) < 1e-9


def test_scheffe_and_johansen():
    s = scheffeci([G1, G2, G3], [1, -0.5, -0.5])
    m = [sum(g) / len(g) for g in (G1, G2, G3)]
    assert abs(s["estimate"] - (m[0] - (m[1] + m[2]) / 2)) < 1e-14
    assert abs(s["S"] - math.sqrt(2 * qf(0.95, 2, 36) * s["mswg"] * (1 / 13 + 0.25 / 15 + 0.25 / 11))) < 1e-13
    assert abs(s["S"] - 5.2262950221183884) < 1e-12
    j = johanq([G1, G2, G3, G4], [[1, 1, -1, -1]])
    xb = [_tmean(g, 0.2) for g in (G1, G2, G3, G4)]
    v = [_yuen_d(g, 0.2)[0] for g in (G1, G2, G3, G4)]
    assert abs(j["statistic"] - (xb[0] + xb[1] - xb[2] - xb[3]) ** 2 / sum(v)) < 1e-15  # one contrast: Q is a squared z
    assert (
        abs(j["statistic"] - 0.0034599116791369511) < 1e-15 and abs(j["crit"] - 4.2118872408292969) < 1e-12
    )  # WRS2:::johan
    assert abs(j["p_value"] - 0.954) <= 0.001  # WRS2::t2way grid p-value
    c = qchisq(1 - j["p_value"], 1)
    assert abs(c + c / 2 * j["A"] * (1 + 3 * c / 3) - j["statistic"]) < 1e-9
    j2 = johanq([G1, G2, G3, G4], [[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]])
    assert abs(j2["statistic"] - 53.084906413588847) < 1e-10 and abs(j2["crit"] - 10.2984362934013) < 1e-10
