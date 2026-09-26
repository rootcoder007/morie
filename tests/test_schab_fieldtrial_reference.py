"""Papadakis (Sec. 6.1.3.2) and the first-difference model (eqs 6.12-6.13) against R lm()."""

import math

from morie.fn.fdiffm import first_difference_model
from morie.fn.papadk import papadakis_analysis

R = [i for i in range(1, 7) for _ in range(4)]
C = [j for _ in range(6) for j in range(1, 5)]
T = [["A", "B", "C", "D"][(i + 2 * j) % 4] for i, j in zip(R, C)]
EFF = {"A": 0.0, "B": 1.2, "C": -0.5, "D": 0.7}
Z = [10 + 0.8 * i - 0.3 * j + EFF[t] + 0.4 * math.sin(3.7 * k) for k, (i, j, t) in enumerate(zip(R, C, T))]


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_papadakis_matches_lm():
    p = papadakis_analysis(Z, R, C, T)
    got = p["treatment_effects"] + p["beta_neighbour"] + p["se_treatment"] + p["se_neighbour"] + [p["sigma2"]]
    # lm(z ~ factor(trt) + x1 + x2) with x1, x2 the E-W and N-S neighbour means of the residuals
    ref = [
        0.84900211261289049,
        -0.20123280111187125,
        0.61809344815996758,
        0.65113055913175832,
        0.41483055271284214,
        0.41958878688856999,
        0.29090510915199419,
        0.40707815523031105,
        0.23114763688194045,
        0.28635972629784023,
        0.25217108577785746,
    ]
    assert all(close(a, b) for a, b in zip(got, ref))
    pc = papadakis_analysis(Z, R, C, T, combined=True)
    got = pc["treatment_effects"] + pc["beta_neighbour"] + [pc["sigma2"]]
    ref = [1.0950737125583965, -0.18587497750231208, 0.88244909427486995, 1.1109653644174815, 0.24517980352926533]
    assert all(close(a, b) for a, b in zip(got, ref))


def test_first_difference_matches_lm():
    f = first_difference_model(Z, T, column=C, row=R)
    # lm(dz ~ dX - 1), differences within each column in row order
    ref = [1.0956879781041891, -0.14878075046031239, 0.81933491301455141]
    se = [0.40344155347946675, 0.44019082316910263, 0.3521526585352821]
    assert all(close(a, b) for a, b in zip(f["tau"], ref))
    assert all(close(a, b) for a, b in zip(f["se"], se))
    assert close(f["sigma2"], 0.93008621185100282) and f["df"] == 17
