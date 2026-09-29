"""Tests for morie.fn.glmpoi: the Poisson score equations at the fit."""

import math

from morie.fn.glmpoi import glmpoi

X = [[k / 5.0, math.cos(k)] for k in range(20)]
Y = [int(abs(math.sin(1.7 * k)) * 4 + k / 5) for k in range(20)]


def test_score_zero_and_deviance():
    r = glmpoi(X, Y)
    Xa = [[1.0] + x for x in X]
    mu = r["fitted"]
    for j in range(3):
        assert abs(sum(Xa[i][j] * (Y[i] - mu[i]) for i in range(20))) < 1e-9
    dev = 2 * sum((y * math.log(y / m) if y > 0 else 0.0) - (y - m) for y, m in zip(Y, mu))
    assert abs(r["deviance"] - dev) < 1e-10
    ll = sum(y * math.log(m) - m - math.lgamma(y + 1) for y, m in zip(Y, mu))
    assert abs(r["aic"] - (-2 * ll + 6)) < 1e-9
