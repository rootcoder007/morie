"""Tests for warpL: the WARP sampled-rank loss and its one-sample step."""

import pytest

from morie.fn.warpL import (
    alpha_weights,
    estimate_rank,
    rank_weight,
    sample_violation,
    warp,
    warp_loss,
    warp_step,
)


class _FixedRng:
    """The only interface warpL asks of an rng: ``uniform()``."""

    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def uniform(self):
        v = self._values[self._i % len(self._values)]
        self._i += 1
        return v


def test_alpha_weights_and_rank_weight():
    """alpha_j = 1/j decreasing; L(r) is its running sum."""
    a = alpha_weights(5)
    assert a == [1.0, 0.5, 1.0 / 3.0, 0.25, 0.2]
    assert alpha_weights(4, "uniform") == [1.0, 1.0, 1.0, 1.0]
    assert alpha_weights(4, "top1") == [1.0, 0.0, 0.0, 0.0]

    assert rank_weight(0, a) == 0.0
    assert rank_weight(1, a) == pytest.approx(1.0, rel=1e-12)
    assert rank_weight(3, a) == pytest.approx(1.0 + 0.5 + 1.0 / 3.0,
                                              rel=1e-12)
    # past the end of alphas the weight saturates
    assert rank_weight(99, a) == pytest.approx(sum(a), rel=1e-12)

    # the point of the reciprocal scheme: rank 1 costs proportionally
    # more than rank 5 than it does under the uniform scheme.
    u = alpha_weights(5, "uniform")
    assert rank_weight(1, a) / rank_weight(5, a) > (
        rank_weight(1, u) / rank_weight(5, u))
    # top1 only ever charges for the very top
    t = alpha_weights(5, "top1")
    assert rank_weight(1, t) == rank_weight(5, t) == 1.0

    with pytest.raises(ValueError):
        alpha_weights(0)
    with pytest.raises(ValueError):
        alpha_weights(3, "quadratic")
    with pytest.raises(ValueError):
        rank_weight(-1, a)


def test_estimate_rank_is_floor_of_labels_over_draws():
    """floor((Y-1)/N): first-draw violation means a badly ranked positive."""
    assert estimate_rank(1, 100) == 99
    assert estimate_rank(2, 100) == 49
    assert estimate_rank(99, 100) == 1
    assert estimate_rank(100, 100) == 0
    # monotone non-increasing in the number of draws
    ranks = [estimate_rank(n, 50) for n in range(1, 50)]
    assert all(ranks[i] >= ranks[i + 1] for i in range(len(ranks) - 1))

    with pytest.raises(ValueError):
        estimate_rank(0, 10)
    with pytest.raises(ValueError):
        estimate_rank(3, 1)


def test_sample_violation_reports_the_cap_separately_from_rank_zero():
    """A cap hit and a rank of 0 are different facts."""
    # negative 0 scores 5.0 and is drawn first, so it violates immediately
    scores = [5.0, -9.0, -9.0, -9.0]
    v = sample_violation(1.0, lambda j: scores[j], 5,
                         _FixedRng([0.0]), margin=1.0)
    assert v["violated"] is True
    assert v["draws"] == 1
    assert v["negative"] == 0
    assert v["negative_score"] == 5.0
    assert v["capped"] is False
    assert v["estimated_rank"] == estimate_rank(1, 5) == 4

    # nothing beats score_positive - margin: the cap is reached and said so
    quiet = sample_violation(1.0, lambda j: -9.0, 5,
                             _FixedRng([0.0]), margin=1.0)
    assert quiet["violated"] is False
    assert quiet["capped"] is True
    assert quiet["draws"] == 4          # Y - 1
    assert quiet["negative"] is None
    assert quiet["estimated_rank"] == 0

    # an explicit cap is honoured
    short = sample_violation(1.0, lambda j: -9.0, 5,
                            _FixedRng([0.0]), margin=1.0, max_draws=2)
    assert short["draws"] == 2 and short["capped"] is True

    # rng cycling over three distinct draws: the third candidate wins
    hit = sample_violation(0.0, lambda j: 1.0 if j == 2 else -9.0, 5,
                           _FixedRng([0.0, 0.25, 0.5]), margin=1.0)
    assert hit["violated"] is True and hit["negative"] == 2
    assert hit["draws"] == 3
    assert hit["estimated_rank"] == estimate_rank(3, 5) == 1

    with pytest.raises(ValueError):
        sample_violation(1.0, lambda j: 0.0, 5, _FixedRng([0.0]),
                         max_draws=0)


def test_warp_loss_is_the_rank_weight_times_the_hinge():
    a = alpha_weights(5)
    out = warp_loss(1.0, 0.5, 3, a, margin=1.0)
    hinge = 1.0 - 1.0 + 0.5
    assert out["hinge"] == pytest.approx(hinge, rel=1e-12)
    assert out["rank_weight"] == pytest.approx(rank_weight(3, a), rel=1e-12)
    assert out["loss"] == pytest.approx(rank_weight(3, a) * hinge, rel=1e-12)
    assert out["estimated_rank"] == 3

    # a satisfied margin clips to exactly zero, whatever the rank
    safe = warp_loss(5.0, 0.0, 4, a, margin=1.0)
    assert safe["hinge"] == 0.0 and safe["loss"] == 0.0

    # the same violation costs more at a worse rank
    assert warp_loss(1.0, 0.5, 4, a)["loss"] > warp_loss(1.0, 0.5, 1, a)["loss"]


def test_warp_step_moves_the_user_toward_the_positive():
    """new_u = u + L(rank) * lr * (positive - negative)."""
    u = [1.0, 0.5, -0.25]
    P = [1.0, 0.0, 0.0]
    negatives = [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    a = alpha_weights(5)
    lr = 0.05

    res = warp_step(P, negatives, u, _FixedRng([0.0]), a, lr=lr, margin=1.0)
    assert res["updated"] is True
    assert res["draws"] == 1
    assert res["negative"] == 0
    assert res["estimated_rank"] == estimate_rank(1, 5)

    sp = sum(x * y for x, y in zip(u, P))            # 1.0
    sn = sum(x * y for x, y in zip(u, negatives[0]))  # 2.0
    ref = warp_loss(sp, sn, estimate_rank(1, 5), a, margin=1.0)
    assert res["rank_weight"] == pytest.approx(ref["rank_weight"], rel=1e-12)
    assert res["loss"] == pytest.approx(ref["loss"], rel=1e-12)
    assert res["estimate"] == pytest.approx(ref["loss"], rel=1e-12)

    g = ref["rank_weight"] * lr
    expected = [u[i] + g * (P[i] - negatives[0][i]) for i in range(3)]
    for got, want in zip(res["user"], expected):
        assert got == pytest.approx(want, rel=1e-12, abs=1e-15)

    # the step is proportional to the rank weight: a top-1 alpha scheme
    # charges less for this rank-4 violation, so it moves less.
    small = warp_step(P, negatives, u, _FixedRng([0.0]),
                      alpha_weights(5, "top1"), lr=lr, margin=1.0)
    assert abs(small["user"][0] - u[0]) < abs(res["user"][0] - u[0])

    # nothing violates: no update, zero loss, the user vector unchanged
    quiet = warp_step(P, [[0.0, 0.0, 0.0]] * 4, u, _FixedRng([0.0]), a,
                      lr=lr, margin=1.0)
    assert quiet["updated"] is False
    assert quiet["loss"] == 0.0
    assert quiet["draws"] == 4
    assert list(quiet["user"]) == u

    # `warp` is the public alias of warp_step
    again = warp(P, negatives, u, _FixedRng([0.0]), a, lr=lr, margin=1.0)
    assert again["loss"] == pytest.approx(res["loss"], rel=1e-12)
