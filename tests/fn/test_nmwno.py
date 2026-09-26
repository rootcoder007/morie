"""wnominate: the W-NOMINATE likelihood recomputed term by term."""

import math

import pytest

from morie.fn import _stats_core as stats
from morie.fn.nmwno import wnominate

VOTES = [[1, 0, 1], [0, 1, float("nan")], [1, 1, 0]]
X = [[-0.6, 0.2], [0.5, -0.1], [0.1, 0.4]]
ZY = [[-0.5, 0.0], [0.4, 0.3], [0.0, -0.4]]
ZN = [[0.5, 0.1], [-0.4, -0.3], [0.2, 0.6]]


def test_loglik_gmp_and_classification_from_first_principles():
    beta, w = 15.0, [1.0, 0.5]
    ll = 0.0
    n = correct = 0
    for i in range(3):
        for j in range(3):
            v = VOTES[i][j]
            if v != v:
                continue
            dy = sum((w[k] * (X[i][k] - ZY[j][k])) ** 2 for k in range(2))
            dn = sum((w[k] * (X[i][k] - ZN[j][k])) ** 2 for k in range(2))
            u = beta * (math.exp(-dy / 2) - math.exp(-dn / 2))
            p = stats.norm.cdf(u)
            ll += math.log(p if v == 1 else 1 - p)
            n += 1
            correct += int((p > 0.5) == (v == 1))
    r = wnominate(VOTES, X, ZY, ZN, beta=beta, w=w)
    assert abs(r["loglik"] - ll) <= 1e-9 * abs(ll)
    assert abs(r["GMP"] - math.exp(ll / n)) <= 1e-12
    assert r["n_total"] == n == 8
    assert r["correct_classification"] == correct / n


def test_docstring_example():
    r = wnominate([[1, 0], [0, 1]], [-0.5, 0.5], [-0.5, 0.5], [0.5, -0.5])
    assert r["correct_classification"] == 1.0


def test_far_tail_vote_has_finite_loglik():
    # a Nay at an overwhelming Yea margin: Phi(-u) underflows 1 - Phi(u)
    r = wnominate([[0]], [[0.0]], [[0.0]], [[5.0]], beta=80.0)
    assert math.isfinite(r["loglik"]) and r["loglik"] < -1000


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        wnominate([[1, 0]], [0.0], [0.0], [0.0])
