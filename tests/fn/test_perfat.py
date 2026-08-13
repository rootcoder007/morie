"""Tests for perfat. Full anchor: ledger/wave3/anchor_nlp_family.py."""
import math
import pytest
from morie.fn import _array_core as np
from morie.fn import _s03core as k
from morie.fn.perfat import (draw_projections, favor_attention,
                             favor_features, kernel_estimate,
                             softmax_attention)

D = 4
X = [0.3, -0.2, 0.5, 0.1]
Y = [-0.1, 0.4, 0.2, -0.3]


def test_the_positive_map_is_unbiased_for_the_softmax_kernel():
    """Lemma 1: E[phi(x)'phi(y)] = exp(x'y), exactly."""
    truth = math.exp(sum(X[i] * Y[i] for i in range(D)))
    est = [kernel_estimate(X, Y, draw_projections(16, D, seed=s,
                                                  orthogonal=False))
           for s in range(300)]
    se = k.sd(est) / math.sqrt(len(est))
    assert abs(k.mean(est) - truth) < 3.0 * se
    assert all(v > 0.0 for v in favor_features(
        [X], draw_projections(16, D, seed=1))[0])


def test_the_trig_map_is_also_unbiased_but_unusable_near_zero():
    """Unbiasedness is NOT what separates them -- the variance is."""
    a, b = [3.0, 0.0, 0.0, 0.0], [-3.0, 0.0, 0.0, 0.0]
    vp, vt = [], []
    for s in range(200):
        om = draw_projections(16, D, seed=1000 + s, orthogonal=False)
        vp.append(kernel_estimate(a, b, om, kind="positive"))
        vt.append(kernel_estimate(a, b, om, kind="trig"))
    assert k.sd(vt) > 100.0 * k.sd(vp)
    assert min(vt) < 0.0          # breaks the convex combination
    assert min(vp) > 0.0


def test_orthogonal_features_lower_the_variance_without_bias():
    truth = math.exp(sum(X[i] * Y[i] for i in range(D)))
    vo = [kernel_estimate(X, Y, draw_projections(16, D, seed=2000 + s,
                                                 orthogonal=True))
          for s in range(200)]
    vi = [kernel_estimate(X, Y, draw_projections(16, D, seed=2000 + s,
                                                 orthogonal=False))
          for s in range(200)]
    assert k.sd(vo) < k.sd(vi)
    assert abs(k.mean(vo) - truth) < 3.0 * k.sd(vo) / math.sqrt(len(vo))


def test_the_approximation_converges_as_features_are_added():
    rng = np.random.default_rng(4)
    L = 5
    Q = [[rng.standard_normal() * 0.4 for _ in range(D)]
         for _ in range(L)]
    K = [[rng.standard_normal() * 0.4 for _ in range(D)]
         for _ in range(L)]
    V = [[rng.standard_normal() for _ in range(3)] for _ in range(L)]
    exact = softmax_attention(Q, K, V)
    gaps = []
    for m in (64, 2048):
        ap = favor_attention(Q, K, V, n_features=m, seed=9)["output"]
        gaps.append(max(abs(exact[i][c] - ap[i][c])
                        for i in range(L) for c in range(3)))
    assert gaps[-1] < gaps[0]
    assert gaps[-1] < 0.15


def test_argument_checks():
    with pytest.raises(ValueError):
        draw_projections(0, D)
    with pytest.raises(ValueError):
        favor_features([X], draw_projections(4, D), kind="nope")
    with pytest.raises(ValueError):
        favor_attention([X], [Y, Y], [[1.0], [1.0]])
