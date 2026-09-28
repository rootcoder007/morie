"""hlthacc against ineq 0.2 (Gini, Theil) and SpatialAcc 0.1 (2SFCA, KD2SFCA), plus hand-derived measures.

SpatialAcc's Hansen family recycles the supply vector down the columns of
the distance matrix, so the gravity measure is checked against the
definition instead.
"""

import math

import pytest

from morie.fn._rng import random_uniform
from morie.fn.hlthacc import (
    deprivation_index,
    fca_accessibility,
    gravity_accessibility,
    health_concentration_index,
    nearest_facility,
    radiation_flows,
    spatial_gini,
    theil_decomposition,
)

INEQ = [0.28905566578122205, 0.13685663909645407, 0.16279294454452048]
ACC_2SFCA = [
    0.0061181548891361247,
    0.0017409470752089136,
    0.0062301733146002654,
    0,
    0.007139047118645196,
    0.0038218350647349448,
    0.004489226239391352,
    0.0060926114991637827,
    0.0053433340098822807,
    0.0036742219994083118,
    0.0054116601839718287,
    0.0046623826946903266,
    0.0017273869346733669,
    0.0017273869346733669,
    0.007139047118645196,
    0.0027618393047179851,
    0.0051922120539102511,
    0.001589825119236884,
    0.0060790513586282359,
    0.0043551733146002655,
    0.0026277863799268985,
    0.0026142262393913521,
    0.0028009428352258735,
    0.0033172120539102512,
    0.0036877821399438587,
]
ACC_KD = [
    0.0059382753172658731,
    0.0006632390592284051,
    0.0067637217886524336,
    0,
    0.0059198995186280755,
    0.0044434753477125034,
    0.0036419141273981884,
    0.0050322852778551348,
    0.0043836562335737182,
    0.002056032590965433,
    0.0055274560269190668,
    0.0044225385464896957,
    0.00073196443247694693,
    0.0010018734789090115,
    0.0078345376756291038,
    0.0033375117115858701,
    0.0068457568877725774,
    0.0025274817642470392,
    0.0077509856689676575,
    0.0051108480554974047,
    0.0036959095066541837,
    0.0031074233870908041,
    0.0023379004490593357,
    0.0031145406695803741,
    0.0036687926909934527,
]

_u = [float(v) for v in random_uniform(300, seed=61, stream=0)]
X = [round(5 + 40 * _u[i], 3) for i in range(25)]
D = [[0.5 + 12 * _u[50 + i * 6 + j] for j in range(6)] for i in range(25)]
P = [round(100 + 900 * _u[200 + i]) for i in range(25)]
S = [round(2 + 10 * _u[240 + j]) for j in range(6)]


def test_gini_theil_equal_ineq():
    assert spatial_gini(X).gini == pytest.approx(INEQ[0], abs=1e-15)
    t = theil_decomposition(X)
    assert pytest.approx((INEQ[1], INEQ[2]), abs=1e-15) == (t.T, t.L)


def test_fca_equal_spatialacc_and_gravity_by_definition():
    assert fca_accessibility(S, P, D, 6.0).access == pytest.approx(ACC_2SFCA, abs=1e-15)
    assert fca_accessibility(S, P, D, 6.0, method="KD2SFCA", power=0.5).access == pytest.approx(ACC_KD, abs=1e-15)
    g = gravity_accessibility(S, D, 0.3)
    assert g == pytest.approx([sum(s * math.exp(-0.3 * d) for s, d in zip(S, row)) for row in D], abs=1e-12)


def test_fca_variants_by_hand():
    Sm, Pm, Dm = [10, 5], [100, 200, 100], [[1, 5], [2, 2], [5, 1]]
    assert fca_accessibility(Sm, Pm, Dm, 3.0).access == pytest.approx([1 / 30, 1 / 30 + 1 / 60, 1 / 60], abs=1e-15)
    e = fca_accessibility(Sm, Pm, Dm, 3.0, method="E2SFCA", steps=[(1.5, 1.0), (3.0, 0.5)])
    # R_1 = 10 / (100 + 0.5 * 200), R_2 = 5 / (0.5 * 200 + 100)
    assert e.ratio == pytest.approx([0.05, 0.025], abs=1e-15)
    three = fca_accessibility(Sm, Pm, Dm, 3.0, method="3SFCA")
    # each demand site with one facility in reach selects it with probability 1
    assert three.access[0] > 0 and three.access[2] > 0


def test_concentration_gini_theil_deprivation_by_hand():
    c = health_concentration_index([4.0, 3.0, 2.0, 1.0], [10, 20, 30, 40])
    assert c.index == pytest.approx(-0.25, abs=1e-15)
    assert c.fractional_rank == [0.125, 0.375, 0.625, 0.875]
    tied = health_concentration_index([1.0, 2.0, 3.0], [5, 5, 9])
    assert tied.fractional_rank == pytest.approx([1 / 3, 1 / 3, 2.5 / 3], abs=1e-15)
    W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    s = spatial_gini([1.0, 2.0, 3.0, 4.0], W)
    assert s.neighbour == pytest.approx(6 / 80, abs=1e-15) and s.gini == pytest.approx(0.25)
    t = theil_decomposition([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1])
    assert t.within + t.between == pytest.approx(t.T, abs=1e-15)
    d = deprivation_index([[1, 2], [3, 2], [5, 8]], method="sum")
    assert d.score == pytest.approx([-1 - 1 / math.sqrt(3), -1 / math.sqrt(3), 1 + 2 / math.sqrt(3)], abs=1e-12)
    tw = deprivation_index([[5, 30, 40, 2], [10, 50, 60, 5], [2, 20, 30, 1]])
    assert tw.z[0][0] == pytest.approx(
        (math.log(6) - (math.log(6) + math.log(11) + math.log(3)) / 3)
        / math.sqrt(
            sum(
                (v - (math.log(6) + math.log(11) + math.log(3)) / 3) ** 2
                for v in (math.log(6), math.log(11), math.log(3))
            )
            / 2
        ),
        abs=1e-12,
    )


def test_nearest_and_radiation():
    assert nearest_facility([[3, 1, 2], [0.5, 4, 4]]).distance == [1.0, 0.5]
    T = radiation_flows([10, 20, 30], [(0, 0), (1, 0), (3, 0)])
    assert T[0][1] == pytest.approx(10 * 10 * 20 / (10 * 30), abs=1e-12)
    assert T[0][2] == pytest.approx(10 * 10 * 30 / (30 * 60), abs=1e-12)


def test_validation():
    with pytest.raises(ValueError):
        fca_accessibility(S, P, D, 6.0, method="E2SFCA")
    with pytest.raises(ValueError):
        deprivation_index([[1, 2, 3]], method="townsend")
