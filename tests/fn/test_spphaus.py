"""Tests for morie.fn.spphaus: the Hausman quadratic form recomputed."""

import math

from morie.fn.spphaus import spphaus


def test_quadratic_form():
    bf, br = [1.2, -0.4, 0.3], [1.0, -0.35, 0.1]
    Vf = [[0.09, 0.01, 0.0], [0.01, 0.04, 0.005], [0.0, 0.005, 0.03]]
    Vr = [[0.05, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.01]]
    d = [a - b for a, b in zip(bf, br)]
    D = [[Vf[i][j] - Vr[i][j] for j in range(3)] for i in range(3)]

    # solve D x = d by Cramer's rule
    def det3(M):
        return (
            M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
        )

    dd = det3(D)
    x = [det3([[d[r] if c == k else D[r][c] for c in range(3)] for r in range(3)]) / dd for k in range(3)]
    h = sum(a * b for a, b in zip(d, x))
    r = spphaus(bf, br, Vf, Vr)
    assert abs(r.statistic - h) < 1e-10
    # chi-square(3) upper tail: erfc(sqrt(h/2)) + sqrt(2h/pi) exp(-h/2)
    assert abs(r.p_value - (math.erfc(math.sqrt(h / 2)) + math.sqrt(2 * h / math.pi) * math.exp(-h / 2))) < 1e-10
