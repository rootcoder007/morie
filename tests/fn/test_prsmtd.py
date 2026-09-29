"""Tests for prsmtd.propensity_score_method."""

from morie.fn import _array_core as np
import math

import pytest

from morie.fn.prsmtd import propensity_score_method


def test_prsmtd_basic():
    rng = np.random.default_rng(42)
    n, T = 300, 3
    H = rng.normal(size=(n, T))
    A = np.zeros((n, T))
    ever = np.zeros(n, dtype=bool)
    for t in range(T):
        start = (rng.random(n) < 0.25) & ~ever
        A[start, t] = 1
        ever |= start
    result = propensity_score_method(A, H)
    m = result["matched_idx"]
    assert m.shape[0] > 0
    for t, i, j in m:
        assert A[i, t] == 1 and A[i, :t].sum() == 0  # newly treated at t
        assert A[j, : t + 1].sum() == 0  # control untreated through t
    # without replacement: no control reused
    assert len({j for _, _, j in m}) == m.shape[0]


def test_prsmtd_edge():
    with pytest.raises(ValueError):
        propensity_score_method([[1, 0]], [[0.5]])  # A/H shape mismatch
    with pytest.raises(ValueError):
        propensity_score_method([[0.5]], [[0.5]])  # non-binary A


def _logit2(x, y):
    """Maximum-likelihood logistic regression of y on (1, x) by Newton's method."""
    b0 = b1 = 0.0
    for _ in range(100):
        p = [1 / (1 + math.exp(-(b0 + b1 * v))) for v in x]
        g0 = sum(a - q for a, q in zip(y, p))
        g1 = sum(v * (a - q) for v, a, q in zip(x, y, p))
        w = [q * (1 - q) for q in p]
        h00, h01, h11 = sum(w), sum(a * v for a, v in zip(w, x)), sum(a * v * v for a, v in zip(w, x))
        det = h00 * h11 - h01 * h01
        s0, s1 = (h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det
        b0, b1 = b0 + s0, b1 + s1
        if max(abs(s0), abs(s1)) < 1e-13:
            break
    return b0, b1


def test_prsmtd_one_period_is_greedy_nearest_propensity_matching():
    """T = 1: logistic MLE propensity on H, initiators in decreasing propensity each take
    the nearest unused control."""
    import math

    A = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    H = [1.2, 0.3, -0.5, 0.2, 0.9, -1.1, 1.0, 0.4, 0.8, -0.2]
    r = propensity_score_method([[a] for a in A], [[h] for h in H])
    b0, b1 = _logit2(H, A)
    e = [1 / (1 + math.exp(-(b0 + b1 * h))) for h in H]
    assert [row[0] for row in r["propensity"].tolist()] == pytest.approx(e, rel=1e-9)
    controls = [i for i in range(10) if A[i] == 0]
    pairs = []
    for i in sorted((i for i in range(10) if A[i]), key=lambda i: -e[i]):
        j = min(controls, key=lambda c: (abs(e[c] - e[i]), controls.index(c)))
        controls.remove(j)
        pairs.append([0, i, j])
    assert r["matched_idx"].tolist() == pairs
