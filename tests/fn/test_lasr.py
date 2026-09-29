"""Tests for lasr: the lasso KKT conditions and the one-predictor soft threshold."""

import math

import pytest

from morie.fn.lasr import lasr

X = [
    [1.0, 0.5, 2.0],
    [2.0, -1.0, 0.1],
    [3.0, 0.2, -1.2],
    [4.0, 1.5, 0.3],
    [5.0, -0.3, 1.1],
    [6.0, 0.8, -0.4],
    [7.0, -1.1, 0.9],
]
Y = [1.1, 2.3, 2.8, 4.4, 4.9, 6.2, 6.8]


@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.4])
def test_lasr_satisfies_the_kkt_conditions(alpha):
    r = lasr(X, Y, alpha=alpha)
    b0, b = float(r["intercept"]), [float(v) for v in r["coef"]]
    n = len(Y)
    res = [y - b0 - math.fsum(c * x for c, x in zip(b, row)) for row, y in zip(X, Y)]
    assert abs(math.fsum(res)) < 1e-10
    for j, bj in enumerate(b):
        g = math.fsum(row[j] * ri for row, ri in zip(X, res)) / n
        if bj != 0.0:
            assert g == pytest.approx(alpha * math.copysign(1.0, bj), abs=1e-6)
        else:
            assert abs(g) <= alpha + 1e-6


def test_lasr_one_predictor_is_the_soft_threshold():
    """b = S(Sxy, n alpha) / Sxx exactly for a single centred predictor."""
    x = [row[0] for row in X]
    n, alpha = len(Y), 0.2
    mx, my = sum(x) / n, sum(Y) / n
    sxx = math.fsum((u - mx) ** 2 for u in x)
    sxy = math.fsum((u - mx) * (v - my) for u, v in zip(x, Y))
    b = math.copysign(max(abs(sxy) - n * alpha, 0.0), sxy) / sxx
    r = lasr([[u] for u in x], Y, alpha=alpha)
    assert float(r["coef"][0]) == pytest.approx(b, rel=1e-12)
    assert float(r["intercept"]) == pytest.approx(my - b * mx, rel=1e-12)
    assert lasr([[u] for u in x], Y, alpha=10.0)["nonzero"] == 0
