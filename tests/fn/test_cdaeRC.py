"""Tests for cdaeRC: collaborative denoising auto-encoder.

Wu, DuBois, Zheng and Ester (2016), eqs. (9)-(13). Each equation is
recomputed by hand here rather than compared with a remembered value.
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.cdaeRC import (cdae, corrupt, decode, encode, fit_cdae,
                             loss, recommend)


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def test_cdaeRC_basic():
    """Eq. (9) mask-out corruption: zero or scaled by 1/(1-q)."""
    y = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    q = 0.25
    yt = corrupt(y, q, np.random.default_rng(3))
    for a, b in zip(y, yt):
        assert b == 0.0 or b == pytest.approx(a / (1.0 - q), rel=1e-15)
    assert corrupt(y, 0.0, np.random.default_rng(3)) == y


def test_corruption_is_unbiased():
    """E[y_tilde] = y is what the 1/(1-q) scaling is for."""
    y = [1.0, 2.0, 0.5]
    q, n = 0.4, 20000
    rng = np.random.default_rng(11)
    means = [0.0, 0.0, 0.0]
    for _ in range(n):
        for j, v in enumerate(corrupt(y, q, rng)):
            means[j] += v / n
    for j, v in enumerate(y):
        # each draw is Bernoulli(1-q) * v/(1-q): variance v^2 q/(1-q)
        se = v * math.sqrt(q / (1.0 - q) / n)
        assert abs(means[j] - v) < 5.0 * se


def test_encode_and_decode_match_equations_10_and_11():
    y_t = [2.0, 0.0, 1.0]
    W = [[0.1, -0.2], [0.5, 0.5], [0.3, 0.4]]
    V_u, b = [0.05, -0.1], [0.2, 0.0]
    z = encode(y_t, W, V_u, b)
    for f in range(2):
        pre = b[f] + V_u[f] + sum(W[i][f] * y_t[i] for i in range(3))
        assert z[f] == pytest.approx(_sig(pre), rel=1e-15)
    z_id = encode(y_t, W, V_u, b, activation="identity")
    assert z_id[0] == pytest.approx(0.2 + 0.05 + 0.2 + 0.3, rel=1e-15)
    Wp, bp = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], [0.0, 0.1, -0.2]
    out = decode(z, Wp, bp)
    for i in range(3):
        pre = bp[i] + sum(Wp[i][f] * z[f] for f in range(2))
        assert out[i] == pytest.approx(_sig(pre), rel=1e-15)
    assert set(decode(z, Wp, bp, items=[2])) == {2}


def test_the_four_losses_match_their_closed_forms():
    assert loss(1.0, 0.3, "square") == pytest.approx(0.5 * 0.7 ** 2, rel=1e-15)
    assert loss(-1.0, 0.3, "log") == pytest.approx(math.log(1 + math.exp(0.3)), rel=1e-15)
    assert loss(1.0, 0.3, "hinge") == pytest.approx(0.7, rel=1e-15)
    assert loss(-1.0, -2.0, "hinge") == 0.0
    p = _sig(0.3)
    assert loss(1.0, 0.3, "cross_entropy") == pytest.approx(-math.log(p), rel=1e-12)
    assert loss(0.0, 0.3, "cross_entropy") == pytest.approx(-math.log(1 - p), rel=1e-12)


def test_fit_recommends_the_held_out_item_of_the_users_cluster():
    # users 0-5 prefer items 0-3, users 6-11 prefer items 4-7; user 0
    # has item 3 held out, so it should rank first among its unseen items
    pos = {u: [0, 1, 2, 3] for u in range(6)}
    pos.update({u: [4, 5, 6, 7] for u in range(6, 12)})
    pos[0] = [0, 1, 2]
    m = fit_cdae(pos, n_users=12, n_items=8, k_dim=4, q=0.2, alpha=0.2,
                 iters=200, n_neg=3, seed=1)
    assert m["loss_history"][-1] < m["loss_history"][0]
    rec = recommend(m, pos, 0, 8, top_k=5)
    assert rec["ranking"][0][0] == 3
    assert all(i not in pos[0] for i, _ in rec["ranking"])
    assert rec["n_scored"] == 8 - len(pos[0])
    assert cdae is fit_cdae


def test_cdaeRC_edge():
    with pytest.raises(ValueError, match="q must lie"):
        corrupt([1.0], 1.0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="y = -1"):
        loss(0.0, 0.3, "log")
    with pytest.raises(ValueError, match="y = -1"):
        loss(0.0, 0.3, "hinge")
    with pytest.raises(ValueError, match="loss must be"):
        loss(1.0, 0.3, "absolute")
    with pytest.raises(ValueError, match="at least 1 user, 2 items"):
        fit_cdae({0: [0]}, n_users=1, n_items=1)
