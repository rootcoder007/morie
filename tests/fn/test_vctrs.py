"""Tests for morie.fn.vctrs: weights recomputed from the Philox stream and the method's scale."""

import math

import pytest

from morie.fn._rng import random_normal, random_uniform
from morie.fn.vctrs import weight_init


def _l(v):
    return v.tolist() if hasattr(v, "tolist") else list(v)


def test_uniform_and_normal_methods_are_scaled_philox_draws():
    fi, fo = 4, 3
    u = _l(random_uniform(12, seed=9))
    a = 2.0 * math.sqrt(6.0 / fi)  # he_uniform with gain 2
    W = weight_init(fi, fo, method="he_uniform", gain=2.0, seed=9).value
    assert max(abs(W[i][j] - (-a + 2 * a * u[i * fo + j])) for i in range(fi) for j in range(fo)) < 1e-15
    z = _l(random_normal(12, seed=9))
    s = math.sqrt(2.0 / (fi + fo))
    W = weight_init(fi, fo, method="xavier_normal", seed=9).value
    assert max(abs(W[i][j] - s * z[i * fo + j]) for i in range(fi) for j in range(fo)) < 1e-15


def test_orthogonal_columns_and_expected_variance():
    r = weight_init(6, 3, method="orthogonal", gain=1.5, seed=3)
    W = r.value
    for a in range(3):
        for b in range(3):
            ip = sum(W[i][a] * W[i][b] for i in range(6))
            assert abs(ip - (2.25 if a == b else 0.0)) < 1e-12
    assert abs(r.extra["expected_variance"] - 2.25 / 6) < 1e-15
    wide = weight_init(2, 5, method="orthogonal", seed=3).value
    assert abs(sum(v * v for v in wide[0]) - 1.0) < 1e-12


def test_large_layer_variance_matches_target():
    r = weight_init(200, 150, method="xavier_uniform", seed=1)
    assert abs(r.extra["variance"] / r.extra["expected_variance"] - 1.0) < 0.02
    with pytest.raises(ValueError):
        weight_init(3, 3, method="bad")
