"""Polarization, bargaining, RPS, party positions, dimensionality, roll calls and Wordscores."""

import math

from morie.fn._rng import random_uniform
from morie.fn.rpscore import ranked_probability_score
from morie.fn.svbarg import spatial_bargaining
from morie.fn.svdimn import dimensionality
from morie.fn.svparty import party_positions
from morie.fn.svpolr import polarization_measures
from morie.fn.svroll import optimal_cutting_lines
from morie.fn.svwords import wordscores


def test_polarization_measures():
    assert polarization_measures([0, 0, 1, 1])["esteban_ray"] == 0.25  # 2 * (1/2)^2 * (1/2) * 1
    er = polarization_measures([0, 1, 1, 3], alpha=0.5)["esteban_ray"]
    p = {0: 0.25, 1: 0.5, 3: 0.25}
    assert abs(er - sum(p[a] ** 1.5 * p[b] * abs(a - b) for a in p for b in p)) < 1e-15
    assert polarization_measures([5, 5, 5, 5])["wolfson"] == 0.0
    x = [1.0, 2.0, 2.5, 4.0, 7.0, 3.0]
    r = polarization_measures(x)
    sv, n = sorted(x), 6
    mu, med = sum(x) / n, 0.5 * (sv[2] + sv[3])
    G = sum(abs(a - b) for a in x for b in x) / (2 * n * n * mu)
    assert abs(r["wolfson"] - 2 * (2 * (0.5 - sum(sv[:3]) / sum(x)) - G) * mu / med) < 1e-15
    g = [0, 0, 0, 1, 1, 1]
    r = polarization_measures(x, party=g)
    mx, mg = sum(x) / 6, 0.5
    cor = sum((a - mx) * (b - mg) for a, b in zip(x, g)) / math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - mg) ** 2 for b in g)
    )
    assert abs(r["sorting"] - cor) < 1e-15 and abs(r["separation"] - 1.8070796117392958) < 1e-12  # R: cohen d pooled
    r = polarization_measures([[0, 1], [1, 0], [0.5, 0.2], [3, 3], [2.5, 4], [4, 2]], party=g)
    assert abs(r["separation"] - 29.29277499088584) < 1e-9  # R: sqrt(mahalanobis(...))
    assert polarization_measures([1, 2], in_rating=[80, 90], out_rating=[20, 30])["affective"] == 60.0


def test_bargaining():
    assert [round(v, 12) for v in spatial_bargaining([0.0], [1.0])["nash"]] == [0.5]
    r = spatial_bargaining([0, 0], [2, 0], status_quo=[1.8, 1.0])
    L, d1, d2 = 4.0, -(1.8**2 + 1), -(0.2**2 + 1)
    grid = max(((-t * t * L - d1) * (-((1 - t) ** 2) * L - d2), t) for t in (k / 200000 for k in range(200001)))
    assert abs(r["nash_t"] - grid[1]) < 1e-5 and r["nash_product"] >= grid[0] - 1e-12
    s = spatial_bargaining(0.0, 1.0, delta1=0.9, delta2=0.8)
    assert abs(s["rubinstein_share"] - 0.2 / 0.28) < 1e-15 and abs(s["rubinstein"][0] - (1 - 0.2 / 0.28)) < 1e-15


def test_rps_and_party_positions():
    assert ranked_probability_score([[0.2, 0.5, 0.3]], [1])["rps"] == [0.065]
    assert ranked_probability_score([[1, 0, 0], [0, 0, 1]], [0, 0])["rps"] == [0.0, 1.0]
    r = party_positions([1, 2, 3, 7, 8, 9], ["D", "D", "D", "R", "R", "R"], n_boot=50, seed=3)
    assert r["position"] == [[2.0], [8.0]] and r["distance"][0][1] == 6.0
    assert all(
        abs(a - b) < 1e-12 for a, b in zip([s[0] for s in r["se"]], [0.47063425145165094, 0.44817594536580735])
    )  # R arm
    assert party_positions([[1, 5], [3, 1], [2, 2]], ["A", "A", "A"], statistic="median", n_boot=0)["position"] == [
        [2.0, 2.0]
    ]


def test_roll_calls_cutting_lines_are_optimal():
    U = [float(u) for u in random_uniform(400, seed=4, stream=0)]
    P = [[2 * U[2 * i] - 1, 2 * U[2 * i + 1] - 1] for i in range(15)]
    s = optimal_cutting_lines(P, n_votes=6, beta=4.0, seed=2)
    brute = []
    for j in range(6):
        col = [s["votes"][i][j] for i in range(15)]
        best = 99
        for k in range(720):
            th = math.pi * k / 360
            pr = sorted((p[0] * math.cos(th) + p[1] * math.sin(th), v) for p, v in zip(P, col))
            best = min(best, min(sum(1 for i, (pv, v) in enumerate(pr) if (i >= c) != (v == 1)) for c in range(16)))
        brute.append(best)
    assert s["errors"] == brute == [0, 0, 0, 1, 1, 1] and s["apre"] == 0.875
    for j in range(6):  # the reported line reproduces its error count
        e = sum(
            (sum(a * b for a, b in zip(P[i], s["normals"][j])) > s["cuts"][j]) != (s["votes"][i][j] == 1)
            for i in range(15)
        )
        assert e == s["errors"][j]
    one = optimal_cutting_lines([[-1.0], [-0.5], [0.4], [1.0]], votes=[[0], [0], [1], [1]])
    assert one["errors"] == [0] and one["apre"] == 1.0
    P3 = [[2 * U[100 + 3 * i] - 1, 2 * U[101 + 3 * i] - 1, 2 * U[102 + 3 * i] - 1] for i in range(12)]
    s3 = optimal_cutting_lines(P3, n_votes=3, beta=3.0, seed=5)
    for j in range(3):  # no plane through the data does better (exhaustive over triples)
        col = [s3["votes"][i][j] for i in range(12)]
        assert s3["errors"][j] <= min(sum(1 for v in col if v == 1), sum(1 for v in col if v == 0))


def test_dimensionality_and_wordscores():
    r = dimensionality([[1, 2], [2, 4], [3, 6], [4, 8]], n_sim=0)
    assert [round(v, 12) for v in r["share"]] == [1.0, 0.0]
    U = [float(u) for u in random_uniform(400, seed=4, stream=0)]
    P = [[2 * U[2 * i] - 1, 2 * U[2 * i + 1] - 1] for i in range(15)]
    votes = optimal_cutting_lines(P, n_votes=6, beta=4.0, seed=2)["votes"]
    d = dimensionality([[float(v) for v in r] for r in votes], n_sim=50, seed=1)
    assert [round(v, 4) for v in d["eigenvalues"]] == [0.5864, 0.2094, 0.1593, 0.1429, 0.0759, 0.0358]
    assert d["elbow"] == 1 and d["parallel"] == 1 and abs(sum(d["share"]) - 1) < 1e-15
    w = wordscores(
        [[10, 5, 0, 0, 5], [0, 5, 5, 0, 10], [0, 0, 5, 10, 5]], [-1.5, 0.0, 1.5], [[5, 5, 5, 5, 5], [8, 4, 1, 0, 7]]
    )
    # word 1 appears only in the first reference: S_w = -1.5; word 2 splits between refs 1 and 2 equally
    assert w["word_scores"][0] == -1.5 and abs(w["word_scores"][1] - (-0.75)) < 1e-15
    assert abs(w["raw"][1] - (8 * -1.5 + 4 * -0.75 + 1 * 0.75 + 0 + 7 * 0.0) / 20) < 1e-15
    assert abs(w["se"][0] - 0.21213203435596426) < 1e-15 and w["rescaled"] == [0.7044101717798211, -1.4169101717798211]
