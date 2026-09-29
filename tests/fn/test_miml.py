"""Tests for morie.fn.miml: Moran's mi of model residuals with randomisation moments."""

from morie.fn.miml import miml
from morie.fn.mirand import mirand

N = 8
W = [[0.5 if abs(i - j) == 1 else 0.0 for j in range(N)] for i in range(N)]
W[0][1] = W[N - 1][N - 2] = 1.0
E = [0.3, -0.2, 0.5, -0.9, 0.1, 0.4, -0.6, 0.35]


def test_residual_moran():
    n = N
    m = sum(E) / n
    z = [v - m for v in E]
    s0 = sum(sum(r) for r in W)
    mi = n / s0 * sum(z[i] * W[i][j] * z[j] for i in range(n) for j in range(n)) / sum(v * v for v in z)
    r = miml(E, W)
    assert abs(r.statistic - mi) < 1e-13
    assert abs(r.expected + 1 / (n - 1)) < 1e-15
    assert abs(r.variance - mirand(E, W).variance) < 1e-15
