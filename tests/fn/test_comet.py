"""Tests for comet: the learned MT metric of Rei, Stewart, Farinha and
Lavie (2020). Features and heads are recomputed by hand."""

import math

import pytest

from morie.fn.comet import (comet, estimator_score, kendall_tau,
                            pooled_features, reference_free, triplet_loss)

H, S, R = [1.0, -2.0, 0.5], [0.5, 1.0, 0.5], [2.0, -1.0, 0.0]


def _features(h, s, r):
    d = range(len(h))
    return (list(h) + list(r) + [h[i] * r[i] for i in d]
            + [abs(h[i] - r[i]) for i in d]
            + [h[i] * s[i] for i in d] + [abs(h[i] - s[i]) for i in d])


def test_comet_basic():
    """Pooled features: hyp, ref, products and |differences| with both."""
    f = pooled_features(H, S, R)
    assert f["features"] == _features(H, S, R)
    assert f["dim"] == 18
    assert f["hyp_ref_diff"] == [1.0, 1.0, 0.5]
    assert f["hyp_src_diff"] == [0.5, 3.0, 0.0]


def test_estimator_head_is_the_linear_map_of_the_features():
    f = _features(H, S, R)
    W = [[0.1 * (j % 5) - 0.2 for j in range(18)]]
    want = 0.3 + sum(W[0][j] * f[j] for j in range(18))
    got = estimator_score(H, S, R, W, b=[0.3])
    assert got["score"] == pytest.approx(want, rel=1e-14)
    assert got["estimate"] == got["score"]
    two = estimator_score(H, S, R, W + W, b=[0.0, 1.0])
    assert two["score"][1] - two["score"][0] == pytest.approx(1.0, rel=1e-14)
    assert comet is estimator_score


def test_triplet_loss_is_two_hinges_on_euclidean_distances():
    better, worse = [1.0, 0.0], [3.0, 0.0]
    src, ref = [0.0, 0.0], [1.0, 1.0]
    d = lambda a, b: math.dist(a, b)
    want_s = max(0.0, d(better, src) - d(worse, src) + 1.0)
    want_r = max(0.0, d(better, ref) - d(worse, ref) + 1.0)
    out = triplet_loss(better, worse, src, ref, margin=1.0)
    assert out["source_term"] == pytest.approx(want_s, rel=1e-14)
    assert out["reference_term"] == pytest.approx(want_r, rel=1e-14)
    assert out["loss"] == pytest.approx(want_s + want_r, rel=1e-14)
    far = triplet_loss([0.0, 0.0], [9.0, 9.0], [0.0, 0.0], [0.0, 0.0])
    assert far["loss"] == 0.0 and far["satisfied"]


def test_kendall_tau_is_the_wmt_relative_ranking_form():
    # (concordant - discordant) / (concordant + discordant), ties dropped
    scores = [0.9, 0.1, 0.5, 0.5, 0.3]
    human = [3.0, 1.0, 2.0, 2.0, 2.0]
    c = d = 0
    for i in range(5):
        for j in range(i + 1, 5):
            a, b = scores[i] - scores[j], human[i] - human[j]
            if a and b:
                c, d = (c + 1, d) if (a > 0) == (b > 0) else (c, d + 1)
    out = kendall_tau(scores, human)
    assert (out["concordant"], out["discordant"]) == (c, d)
    assert out["tau"] == pytest.approx((c - d) / (c + d), rel=1e-15)
    assert kendall_tau([1, 2, 3], [3, 2, 1])["tau"] == -1.0


def test_reference_free_uses_source_features_only():
    d = range(3)
    f = H + S + [H[i] * S[i] for i in d] + [abs(H[i] - S[i]) for i in d]
    W = [[0.5] * 12]
    out = reference_free(H, S, W, b=[-1.0])
    assert out["score"] == pytest.approx(-1.0 + 0.5 * sum(f), rel=1e-14)
    assert out["reference_used"] is False


def test_comet_edge():
    with pytest.raises(ValueError, match="differ in length"):
        pooled_features([1.0], [1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="expects 5 features"):
        estimator_score(H, S, R, [[0.0] * 5])
    with pytest.raises(ValueError, match="margin"):
        triplet_loss(H, S, S, R, margin=0.0)
    with pytest.raises(ValueError, match="at least 2 segments"):
        kendall_tau([1.0], [1.0])
    with pytest.raises(ValueError, match="human judgements"):
        kendall_tau([1.0, 2.0], [1.0])
