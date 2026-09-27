"""Urban indicators: accessibility (SpatialAcc 2SFCA), coverage, LILA, fragmentation (landscapemetrics), sprawl, NDISI, SUHI, density."""

import math

from morie.fn._rng import random_uniform
from morie.fn.accidx import spatial_accessibility
from morie.fn.fooddes import food_desert
from morie.fn.fragmt import landscape_fragmentation
from morie.fn.imperv import impervious_indices
from morie.fn.popden import population_density_surface
from morie.fn.servcov import service_coverage
from morie.fn.sprawl import sprawl_entropy
from morie.fn.uhiint import uhi_intensity

PINS = {
    "2sfca": [
        0.02394247703231741,
        0.0,
        0.015899581589958158,
        0.025130890052356022,
        0.02394247703231741,
        0.00804289544235925,
        0.0,
        0.06029864051704925,
        0.03514090006236603,
        0.00804289544235925,
        0.038756724447101015,
        0.07030865052705926,
        0.025130890052356022,
        0.07030865052705926,
        0.03514090006236603,
        0.07030865052705926,
        0.015899581589958158,
        0.0,
        0.0,
        0.02394247703231741,
        0.038756724447101015,
        0.0,
        0.07258239772227243,
        0.00804289544235925,
        0.0,
    ],
    "g2sfca": [
        0.029150230387718974,
        0.0,
        0.023267125671903338,
        0.013609130249552648,
        0.020330054700664558,
        0.008223695768721928,
        0.0,
        0.05624743747533803,
        0.015832856231617225,
        0.003674361825801705,
        0.02028125037449722,
        0.08841857565183607,
        0.004377737180963742,
        0.09845486713139111,
        0.050237499488007806,
        0.08472063535221562,
        0.0272642646169539,
        0.0,
        0.0,
        0.02179406287053536,
        0.03199711403895306,
        0.0,
        0.06288954399849697,
        0.009798445026310392,
        0.0,
    ],
    "hansen": [
        14.832313357579608,
        4.509531502127556,
        13.091578719869979,
        16.914597331845194,
        11.861786501737178,
        5.494724545575256,
        7.0070494132511625,
        25.018888048936837,
        19.558399284560156,
        7.622233123023184,
        18.263798062579063,
        29.72306760202427,
        11.504030473913252,
        35.29939469811621,
        23.840371643841397,
        26.462428182910422,
        16.44723213058611,
        6.087264871078565,
        3.738165006193057,
        17.212503383819897,
        17.562187374653192,
        5.483629172198194,
        27.389859212363344,
        10.627713244124426,
        6.794887188150941,
    ],
    "gravity": [
        0.02541166788228192,
        0.02016231971623179,
        0.024458935021881515,
        0.027098127317088085,
        0.02379533528340932,
        0.02028962286175161,
        0.02182104949785555,
        0.03227472560359422,
        0.02830328962088734,
        0.0211500027307046,
        0.027662317600126794,
        0.03372065761886534,
        0.024363437487666926,
        0.040048904301867486,
        0.030234034470758436,
        0.03177555937687213,
        0.026554414023993944,
        0.02115523815666748,
        0.01861460598182412,
        0.026980229117068998,
        0.027111457508217862,
        0.019825953635055595,
        0.03232688062384652,
        0.023903507166621,
        0.02176885828147356,
    ],
    "cov": 0.7230058989484484,
    "frag": [24.6, 4.878048780487805, 0.795, 2],
}


def town():
    U = [float(u) for u in random_uniform(2000, seed=17, stream=0)]
    dem = [(10 * U[2 * i], 10 * U[2 * i + 1]) for i in range(25)]
    sup = [(10 * U[100 + 2 * j], 10 * U[101 + 2 * j]) for j in range(6)]
    P = [round(50 + 200 * U[200 + i]) for i in range(25)]
    S = [round(5 + 20 * U[300 + j]) for j in range(6)]
    D = [[math.dist(a, b) for b in sup] for a in dem]
    grid = [[1 if U[500 + 12 * y + x] < 0.55 else 0 for x in range(12)] for y in range(10)]
    return P, S, D, grid


def test_accessibility_formulas():
    P, S, D, _ = town()
    for m in ("2sfca", "g2sfca", "hansen", "gravity"):
        a = spatial_accessibility(P, S, D, method=m, d0=3.0, beta=0.5)["accessibility"]
        assert all(abs(u - v) < 1e-12 for u, v in zip(a, PINS[m]))  # 2sfca equals SpatialAcc::ac (R test)
    h = spatial_accessibility(P, S, D, method="hansen", beta=0.5)["accessibility"]
    assert all(abs(h[i] - sum(S[j] * math.exp(-0.5 * D[i][j]) for j in range(6))) < 1e-12 for i in range(25))
    # with every site inside every catchment, 2SFCA gives each location total supply / total demand
    big = spatial_accessibility(P, S, D, d0=100.0)["accessibility"]
    assert all(abs(v - sum(S) / sum(P)) < 1e-15 for v in big)
    # the gravity model preserves total supply: sum_i P_i A_i = sum_j S_j
    g = spatial_accessibility(P, S, D, method="gravity", beta=0.5)["accessibility"]
    assert abs(sum(p * a for p, a in zip(P, g)) - sum(S)) < 1e-9
    n = spatial_accessibility(P, S, D, method="nearest", d0=3.0)
    assert n["nearest"] == [min(r) for r in D] and n["within"] == [sum(v <= 3.0 for v in r) for r in D]


def test_service_coverage_and_food_deserts():
    P, S, D, _ = town()
    c = service_coverage(D, 2.5, P)
    assert abs(c["coverage_share"] - PINS["cov"]) < 1e-15
    assert abs(c["covered_weight"] - sum(p for p, r in zip(P, D) if min(r) <= 2.5)) < 1e-9
    assert service_coverage([[1.0, 5.0], [4.0, 2.5], [6.0, 7.0]], 3.0, [10, 20, 30])["multiplicity"] == [1, 1, 0]
    f = food_desert(
        [4000, 4000, 900, 3000],
        [1500, 100, 400, 1200],
        [0.25, 0.25, 0.10, 0.12],
        [40000, 40000, 70000, 45000],
        [60000] * 4,
        [True, True, False, True],
    )
    assert (
        f["low_income"] == [True, True, False, True]
        and f["low_access"] == [True, False, True, True]
        and f["lila"] == [True, False, False, True]
    )


def test_fragmentation_matches_landscapemetrics():
    _, _, _, grid = town()
    f = landscape_fragmentation(grid)
    assert [f["mesh"], f["splitting"], f["division"], f["n_patches"]] == PINS["frag"]  # 24.6, 4.878049, 0.795 (R test)
    A = 120.0
    assert (
        abs(f["mesh"] - sum(a * a for a in f["patch_areas"]) / A) < 1e-12 and abs(f["splitting"] * f["mesh"] - A) < 1e-9
    )
    four = landscape_fragmentation(grid, eight=False)
    assert four["n_patches"] >= f["n_patches"] and four["mesh"] <= f["mesh"]


def test_sprawl_impervious_uhi_density():
    assert sprawl_entropy([5, 5, 5, 5])["relative"] == 1.0
    assert sprawl_entropy([10, 0, 0])["relative"] == 0.0
    s = sprawl_entropy([4, 2, 1, 1], area=[1, 1, 1, 1])
    assert abs(s["entropy"] - (-0.5 * math.log(0.5) - 0.25 * math.log(0.25) - 2 * 0.125 * math.log(0.125))) < 1e-15
    r = impervious_indices([0.1, 0.05], [0.2, 0.4], [0.3, 0.1], tir=[0.8, 0.02])
    vis = (r["mndwi"][0] + 0.2 + 0.3) / 3
    assert abs(r["ndisi"][0] - (0.8 - vis) / (0.8 + vis)) < 1e-15 and r["impervious"] == [True, False]
    u = uhi_intensity([30, 32, 26, 24, 28], [True, True, False, False, False], rural=[False, False, True, True, False])
    assert u["intensity"] == 6.0 and u["anomaly"][4] == 3.0
    g = [(x / 20 - 3, y / 20 - 3) for x in range(121) for y in range(121)]
    d = population_density_surface([(0, 0), (0.5, 0.2)], [10, 5], g, 1.2)["density"]
    assert abs(sum(d) / 400 - 15) < 1e-3  # the quartic kernel integrates to one


def test_ndisi_uses_the_mean_of_three_bands():
    r = impervious_indices([0.2], [0.2], [0.3], tir=[0.6])
    vis = (-0.2 + 0.2 + 0.3) / 3  # MNDWI = (0.2 - 0.3) / 0.5
    assert abs(r["mndwi"][0] + 0.2) < 1e-15 and abs(r["ndisi"][0] - (0.6 - vis) / (0.6 + vis)) < 1e-15
