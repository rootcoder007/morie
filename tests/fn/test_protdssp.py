import math

import pytest

from morie.fn.protdssp import dssp_assign, dssp_hbond_energy

# ubiquitin (PDB 1UBQ) residues 1-30, backbone N CA C O; sequence MQIFVKTLTGKTITLEVEPSDTIENVKAKI
_UBQ = """
27.340 24.430 2.614 26.266 25.413 2.842 26.913 26.639 3.531 27.886 26.463 4.263
26.335 27.770 3.258 26.850 29.021 3.898 26.100 29.253 5.202 24.865 29.024 5.330
26.849 29.656 6.217 26.235 30.058 7.497 26.882 31.428 7.862 27.906 31.711 7.264
26.214 32.097 8.771 26.772 33.436 9.197 27.151 33.362 10.650 26.350 32.778 11.395
28.260 33.943 11.096 28.605 33.965 12.503 28.638 35.461 12.900 29.522 36.103 12.320
27.751 35.867 13.740 27.691 37.315 14.143 28.469 37.475 15.420 28.213 36.753 16.411
29.426 38.430 15.446 30.225 38.643 16.662 29.664 39.839 17.434 28.850 40.565 16.859
30.132 40.069 18.642 29.607 41.180 19.467 30.075 42.538 18.984 29.586 43.570 19.483
30.991 42.571 17.998 31.422 43.940 17.553 30.755 44.351 16.277 31.207 45.268 15.566
29.721 43.673 15.885 28.978 43.960 14.678 29.604 43.507 13.393 29.219 43.981 12.301
30.563 42.623 13.495 31.191 42.012 12.331 30.459 40.666 12.130 30.253 39.991 13.133
30.163 40.338 10.886 29.542 39.020 10.653 30.494 38.261 9.729 30.849 38.850 8.706
30.795 37.015 10.095 31.720 36.289 9.176 30.955 35.211 8.459 30.025 34.618 9.040
31.244 34.986 7.197 30.505 33.884 6.512 31.409 32.680 6.446 32.619 32.812 6.125
30.884 31.485 6.666 31.677 30.275 6.639 31.022 29.288 5.665 29.809 29.395 5.545
31.834 28.412 5.125 31.220 27.341 4.275 31.440 26.079 5.080 32.576 25.802 5.461
30.310 25.458 5.384 30.288 24.245 6.193 29.279 23.227 5.641 28.478 23.522 4.725
29.380 22.057 6.232 28.468 20.940 5.980 27.819 20.609 7.316 28.449 20.674 8.360
26.559 20.220 7.288 25.829 19.825 8.494 26.541 18.732 9.251 26.333 18.536 10.457
27.361 17.959 8.559 28.054 16.835 9.210 29.258 17.318 9.984 29.930 16.477 10.606
29.599 18.599 9.828 30.796 19.083 10.566 30.491 19.162 12.040 29.367 19.523 12.441
31.510 18.936 12.852 31.398 19.064 14.286 31.593 20.553 14.655 32.159 21.311 13.861
31.113 20.863 15.860 31.288 22.201 16.417 32.776 22.519 16.577 33.233 23.659 16.384
33.548 21.526 16.950 35.031 21.722 17.069 35.615 22.190 15.759 36.532 23.046 15.724
35.139 21.624 14.662 35.590 21.945 13.302 35.238 23.382 12.920 36.066 24.109 12.333
34.007 23.745 13.250 33.533 25.097 12.978 34.441 26.099 13.684 34.883 27.090 13.093
34.734 25.822 14.949 35.596 26.715 15.736 36.975 26.826 15.107 37.579 27.926 15.159
37.499 25.743 14.571 38.794 25.761 13.880 38.728 26.591 12.611 39.704 27.346 12.277
37.633 26.543 11.867 37.471 27.391 10.668 37.441 28.882 11.052 38.020 29.772 10.382
36.811 29.170 12.192 36.731 30.570 12.645 38.148 30.981 13.069 38.544 32.150 12.856
"""
_SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKI"


def _ubq():
    out = []
    for line in _UBQ.strip().splitlines():
        v = [float(x) for x in line.split()]
        out.append([v[0:3], v[3:6], v[6:9], v[9:12]])
    return out


def _energy(n, h, c, o):
    e = 27.888 * (1 / math.dist(o, n) + 1 / math.dist(c, h) - 1 / math.dist(o, h) - 1 / math.dist(c, n))
    return max(e, -9.9)


def _nerf(a, b, c, bond, angle, torsion):
    bc = [c[k] - b[k] for k in range(3)]
    lbc = math.sqrt(sum(x * x for x in bc))
    bc = [x / lbc for x in bc]
    ab = [b[k] - a[k] for k in range(3)]
    n = [ab[1] * bc[2] - ab[2] * bc[1], ab[2] * bc[0] - ab[0] * bc[2], ab[0] * bc[1] - ab[1] * bc[0]]
    ln = math.sqrt(sum(x * x for x in n))
    n = [x / ln for x in n]
    m = [n[1] * bc[2] - n[2] * bc[1], n[2] * bc[0] - n[0] * bc[2], n[0] * bc[1] - n[1] * bc[0]]
    ang = math.radians(angle)
    tor = math.radians(torsion)
    d = [-bond * math.cos(ang), bond * math.sin(ang) * math.cos(tor), bond * math.sin(ang) * math.sin(tor)]
    return [c[k] + d[0] * bc[k] + d[1] * m[k] + d[2] * n[k] for k in range(3)]


def _helix(nres, phi=-57.0, psi=-47.0):
    N = [[0.0, 0.0, 0.0]]
    CA = [[1.458, 0.0, 0.0]]
    t = math.radians(111.2)
    C = [[1.458 - 1.525 * math.cos(t), 1.525 * math.sin(t), 0.0]]
    for i in range(nres - 1):
        N.append(_nerf(N[i], CA[i], C[i], 1.329, 116.2, psi))
        CA.append(_nerf(CA[i], C[i], N[i + 1], 1.458, 121.7, 180.0))
        C.append(_nerf(C[i], N[i + 1], CA[i + 1], 1.525, 111.2, phi))
    OX = [_nerf(N[i], CA[i], C[i], 1.231, 120.5, psi + 180.0) for i in range(nres)]
    return [[N[i], CA[i], C[i], OX[i]] for i in range(nres)]


def test_hbond_energy_formula():
    n, h, c, o = [0.3, -0.2, 0.1], [1.2, 0.1, -0.1], [4.0, 0.5, 0.2], [2.9, 0.3, 0.0]
    assert dssp_hbond_energy(n, h, c, o) == pytest.approx(_energy(n, h, c, o), abs=1e-12)
    assert dssp_hbond_energy([0, 0, 0], [1, 0, 0], [1.9, 0, 0], [1.2, 0, 0]) == -9.9


def test_reported_hbonds_recompute_from_coordinates():
    res = _ubq()
    r = dssp_assign(res, sequence=_SEQ)
    assert len(r.ss) == 30 and set(r.ss) <= set("HBEGITS-")
    for i, bonds in enumerate(r.hbonds):
        if i == 0:
            h = res[0][0]
        else:
            u = [res[i - 1][2][k] - res[i - 1][3][k] for k in range(3)]
            d = math.sqrt(sum(x * x for x in u))
            h = [res[i][0][k] + u[k] / d for k in range(3)]
        assert len(bonds) <= 2
        for acc, e in bonds:
            assert e < -0.5
            assert e == pytest.approx(_energy(res[i][0], h, res[acc][2], res[acc][3]), abs=1e-12)
            assert math.dist(res[i][1], res[acc][1]) < 9.0
    assert r.hbonds[_SEQ.index("P")] == []


def test_assignment_follows_the_definitions():
    res = _ubq()
    r = dssp_assign(res, sequence=_SEQ)

    def bond(d, a):
        return any(x == a for x, _ in r.hbonds[d])

    turn4 = [i + 4 < 30 and bond(i + 4, i) for i in range(30)]
    for i, s in enumerate(r.ss):
        if s == "H":
            assert any(turn4[k] and turn4[k - 1] for k in range(max(1, i - 3), i + 1))
    in_ladder = set()
    for typ, ii, jj in r.bridges:
        in_ladder.update(range(ii[0], ii[-1] + 1))
        in_ladder.update(range(jj[0], jj[-1] + 1))
        i, j = ii[0], jj[-1] if typ == "antiparallel" else jj[0]
        if typ == "antiparallel":
            assert (bond(j, i) and bond(i, j)) or (bond(i + 1, j - 1) and bond(j + 1, i - 1))
        else:
            assert (bond(i + 1, j) and bond(j, i - 1)) or (bond(j + 1, i) and bond(i, j - 1))
    assert {i for i, s in enumerate(r.ss) if s in "EB"} <= in_ladder
    assert r.ss.count("E") >= 8 and r.ss.count("H") >= 4


def test_ideal_alpha_helix_is_all_h_inside():
    res = _helix(16)
    r = dssp_assign(res)
    for i in range(12):
        assert any(a == i for a, _ in r.hbonds[i + 4])
    assert r.ss == "-" + "H" * 14 + "-"
    assert dssp_assign(res, prefer_pi=False).ss == r.ss
