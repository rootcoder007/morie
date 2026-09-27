"""Massey-Denton segregation indices against OasisR (4-decimal output) and closed forms."""

import math

from morie.fn._rng import random_uniform
from morie.fn.centidx import centralization_index
from morie.fn.clusidx import clustering_index
from morie.fn.concidx import spatial_concentration
from morie.fn.dsmidx import dissimilarity_index
from morie.fn.evenidx import segregation_evenness
from morie.fn.expidx import exposure_index
from morie.fn.isoidx import isolation_index


def city():
    U = [float(u) for u in random_uniform(1000, seed=13, stream=0)]
    nr, nc = 5, 6
    n = nr * nc
    xy = [(i % nc + 0.3 * U[2 * i], i // nc + 0.3 * U[2 * i + 1]) for i in range(n)]
    x = [
        [
            round(40 * U[100 + 3 * i] * (1 + (i % nc) / 3)),
            round(30 * U[101 + 3 * i] * (1 + (i // nc) / 2)),
            round(20 * U[102 + 3 * i]) + 1,
        ]
        for i in range(n)
    ]
    area = [0.5 + U[300 + i] for i in range(n)]
    c = [[1 if abs(i % nc - j % nc) + abs(i // nc - j // nc) == 1 else 0 for j in range(n)] for i in range(n)]
    d = [[math.dist(xy[i], xy[j]) for j in range(n)] for i in range(n)]
    return {"x": x, "area": area, "c": c, "d": d, "dc": [math.dist(xy[i], xy[0]) for i in range(n)]}


DATA = city()

OASIS = {
    "D": [[0, 0.4613, 0.3842], [0.4613, 0, 0.3395], [0.3842, 0.3395, 0]],
    "IS": [0.4175, 0.398, 0.2782],
    "Dm": [[0, 0.155, 0.0835], [0.155, 0, 0.0987], [0.0835, 0.0987, 0]],
    "xPy": [[0.5806, 0.2695, 0.1499], [0.3161, 0.5117, 0.1722], [0.3837, 0.3756, 0.2407]],
    "xPyx": [[0.5752, 0.2727, 0.1522], [0.3199, 0.5052, 0.1748], [0.3894, 0.3814, 0.2292]],
    "xPx": [0.5806, 0.5117, 0.2407],
    "eta2": [0.2432, 0.2124, 0.0805],
    "DEL": [0.361, 0.3225, 0.1858],
    "ACO": [0.5686, 0.4589, 0.5201],
    "RCO": [[0, 0.1614, 0.0353], [-0.176, 0, -0.1188], [-0.0324, 0.1011, 0]],
    "ACLc": [0.1522, 0.0853, 0.0086],
    "ACLd": [0.1407, 0.0805, 0.0093],
    "RCL": [[0, 0.0538, 0.151], [-0.0511, 0, 0.0923], [-0.1312, -0.0845, 0]],
    "SP": [1.0891],
    "ACE": [-0.0738, -0.065, 0.0695],
    "RCE": [[0, -0.0119, -0.1523], [0.0119, 0, -0.147], [0.1523, 0.147, 0]],
    "H": [0.1977, 0.1782, 0.0789],
    "atk": [0.2742, 0.2533, 0.1204],
    "gini": [0.5697, 0.5433, 0.3837],
}


def close(a, b, tol=5e-5):  # OasisR rounds to 4 decimals
    if isinstance(a, list):
        return all(close(u, v, tol) for u, v in zip(a, b))
    return abs(a - b) <= tol


def test_evenness_and_exposure_match_oasisr():
    x, c = DATA["x"], DATA["c"]
    r = dissimilarity_index(x, c)
    assert close(r["D"], OASIS["D"]) and close(r["IS"], OASIS["IS"]) and close(r["D_morrill"], OASIS["Dm"])
    assert close(exposure_index(x)["xPy"], OASIS["xPy"]) and close(exposure_index(x, exact=True)["xPy"], OASIS["xPyx"])
    r = isolation_index(x)
    assert close(r["xPx"], OASIS["xPx"]) and close(r["eta2"], OASIS["eta2"])
    r = segregation_evenness(x)
    assert close(r["H"], OASIS["H"]) and close(r["atkinson"], OASIS["atk"]) and close(r["gini"], OASIS["gini"])


def test_concentration_clustering_centralization_match_oasisr():
    x, c, d, a, dc = DATA["x"], DATA["c"], DATA["d"], DATA["area"], DATA["dc"]
    r = spatial_concentration(x, a)
    assert close(r["DEL"], OASIS["DEL"]) and close(r["ACO"], OASIS["ACO"]) and close(r["RCO"], OASIS["RCO"])
    assert close(clustering_index(x, contiguity=c)["ACL"], OASIS["ACLc"])
    r = clustering_index(x, distance=d)
    assert close(r["ACL"], OASIS["ACLd"]) and close(r["RCL"], OASIS["RCL"]) and close(r["SP"], OASIS["SP"][0])
    r = centralization_index(x, dc, a)
    assert close(r["ACE"], OASIS["ACE"]) and close(r["RCE"], OASIS["RCE"])


def test_morgan_distance_decay_uses_destination_populations():
    # OasisR::DPxx weights by t_i (R recycling); Morgan (1983) weights K_ij by t_j
    x, d = DATA["x"], DATA["d"]
    t = [sum(r) for r in x]
    n = len(x)
    K = [[t[j] * math.exp(-d[i][j]) for j in range(n)] for i in range(n)]
    K = [[v / sum(row) for v in row] for row in K]
    tot = [sum(r[k] for r in x) for k in range(3)]
    ref = [sum(x[i][k] / tot[k] * sum(K[i][j] * x[j][k] / t[j] for j in range(n)) for i in range(n)) for k in range(3)]
    assert close(isolation_index(x, distance=d)["DPxx"], ref, 1e-12)
    assert close(
        isolation_index(x, distance=d)["DPxx"], [0.48982610740430754, 0.4211546654088076, 0.18334038623890425], 1e-12
    )


def test_closed_forms():
    assert dissimilarity_index([[10, 0], [0, 10]])["D"][0][1] == 1.0
    assert dissimilarity_index([[3, 6], [5, 10], [2, 4]])["D"][0][1] < 1e-15  # identical composition
    e = exposure_index([[3, 7], [8, 2], [5, 5]])["xPy"]
    assert all(abs(sum(row) - 1) < 1e-15 for row in e)  # a member meets someone of some group
    assert segregation_evenness([[10, 0], [0, 10]])["H"] == [1.0, 1.0]
    # cumulative shares X = (2/3, 1, 1), Y = (0, 1/3, 1): RCE = (2/9 + 1) - 1/3 = 8/9
    assert abs(centralization_index([[10, 0], [5, 5], [0, 10]], [0, 1, 2])["RCE"][0][1] - 8 / 9) < 1e-15
    # DEL = (|1 - 1/4| + |0 - 3/4|) / 2 = 3/4
    assert abs(spatial_concentration([[10, 0], [0, 10]], [1, 3])["DEL"][0] - 0.75) < 1e-15


def test_atkinson_shape_parameter():
    # b = 0.3 separates the exponent 1 / (1 - b) from 1 / b (equal at the default 0.5)
    x = [[12, 3], [4, 9], [7, 7], [1, 10]]
    t = [sum(r) for r in x]
    T = sum(t)
    P = sum(r[0] for r in x) / T
    p = [r[0] / s for r, s in zip(x, t)]
    s = sum((1 - pi) ** 0.7 * pi**0.3 * ti / (P * T) for pi, ti in zip(p, t))
    assert abs(segregation_evenness(x, delta=0.3)["atkinson"][0] - (1 - P / (1 - P) * s ** (1 / 0.7))) < 1e-15
