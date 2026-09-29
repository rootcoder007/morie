"""Tests for morie.fn.carres: Moran's I of CAR residuals."""

from morie.fn.carres import carres

N = 8
W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(N)] for i in range(N)]
E = [0.3, -0.2, 0.5, -0.9, 0.1, 0.4, -0.6, 0.35]


def test_moran_of_residuals():
    m = sum(E) / N
    z = [v - m for v in E]
    s0 = sum(sum(r) for r in W)
    mi = N / s0 * sum(z[i] * W[i][j] * z[j] for i in range(N) for j in range(N)) / sum(v * v for v in z)
    r = carres(E, W)
    assert abs(r.statistic - mi) < 1e-13
    assert abs(r.expected + 1 / (N - 1)) < 1e-15
