"""bayesreg: sampler building blocks, reproducibility and clear-cut posteriors."""

import math

import pytest

from morie.fn._rng import random_normal
from morie.fn.bayesreg import (
    _St,
    bayes_a,
    bayes_b,
    bayes_linear_halfcauchy,
    contaminated_normal_outliers,
    finite_mixture_gibbs,
    genomic_reliability,
)


def test_gamma_draws_have_the_right_moments():
    st = _St(9)
    for shape in (0.4, 1.0, 3.7):
        d = [st.gamma(shape) for _ in range(6000)]
        m = sum(d) / len(d)
        v = sum((x - m) ** 2 for x in d) / (len(d) - 1)
        assert abs(m - shape) < 4 * math.sqrt(shape / len(d))
        assert abs(v - shape) < 0.15 * shape + 0.05


def test_reproducible_and_sensible():
    z = [float(v) for v in random_normal(80, seed=2)]
    X = [[1.0, z[i]] for i in range(30)]
    y = [1 + 2 * z[i] + 0.3 * z[40 + i] for i in range(30)]
    a = bayes_linear_halfcauchy(y, X, ndraw=300, burn_in=50, seed=5)
    b = bayes_linear_halfcauchy(y, X, ndraw=300, burn_in=50, seed=5)
    assert a.draws == b.draws
    assert abs(a.beta[1] - 2) < 0.3
    mix = finite_mixture_gibbs([0.1 * v for v in z[:20]] + [10 + 0.1 * v for v in z[20:40]], 2, ndraw=300, burn_in=50)
    assert abs(mix.mu[0]) < 0.5 and abs(mix.mu[1] - 10) < 0.5 and mix.pi[0] == pytest.approx(0.5, abs=0.2)
    co = contaminated_normal_outliers(z[:20] + [12.0], ndraw=400, burn_in=50)
    assert co.outlier_prob[-1] > 0.95 and max(co.outlier_prob[:20]) < 0.5


def test_marker_models_and_reliability():
    z = [float(v) for v in random_normal(200, seed=3)]
    M = [[float(int(abs(z[5 * i + j]) > 0.6)) for j in range(5)] for i in range(30)]
    y = [2.0 * M[i][0] + 0.2 * z[150 + i] for i in range(30)]
    a = bayes_a(y, M, ndraw=300, burn_in=50, seed=1)
    assert abs(a.effects[0] - 2.0) < 0.4
    b = bayes_b(y, M, pi=0.8, ndraw=300, burn_in=50, seed=2)
    assert b.inclusion[0] > 0.9
    r = genomic_reliability([0.1, 0.4, 0.9], 0.8)
    assert r.reliability == pytest.approx([0.875, 0.5, -0.125]) and r.accuracy[2] == 0.0
