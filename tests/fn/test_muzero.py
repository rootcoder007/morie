"""Tests for muzero.muzero (MCTS with a learned model, Schrittwieser et
al. 2020, Appendix B)."""

import pytest

from morie.fn.muzero import muzero


def _rep(o):
    return 0


def _dyn(s, a):
    # action "L" pays 1, "R" pays 0; the latent state just counts depth
    return (1.0 if a == "L" else 0.0), s + 1


def _pred(s):
    return [0.5, 0.5], 0.0


def test_muzero_basic():
    """Search bookkeeping from eqs. 2-5: one dynamics and one prediction
    call per simulation plus the root prediction; visits sum to the
    simulation count; the root value is sum_a N(a) Q(a) / sum_a N(a);
    the paying action draws the most visits and, at temperature 0, the
    policy is one-hot on it."""
    r = muzero(None, ["L", "R"], _rep, _dyn, _pred, simulations=30, temperature=0)
    assert isinstance(r, dict)
    N, Q = r["visits"], r["Q"]
    assert sum(N.values()) == 30
    assert r["n_dynamics_calls"] == 30 and r["n_prediction_calls"] == 31
    assert r["value"] == pytest.approx(sum(N[a] * Q[a] for a in N) / 30, rel=1e-14)
    assert N["L"] > N["R"] and r["action"] == "L"
    assert list(r["policy"]) == [1.0, 0.0]


def test_muzero_edge():
    """Q(s, a) = r(s, a) + gamma V(child), worked by hand. One simulation
    (prior tie -> first action) expands L: Q(L) = 1 + gamma * 0. The
    second descends L then L again: the grandchild backs up v = 0, so
    V(L) = (0 + 1) / 2 and Q(L) = 1 + gamma / 2. At temperature 1 the
    policy is the visit share."""
    g = 0.997
    r1 = muzero(None, ["L", "R"], _rep, _dyn, _pred, simulations=1, gamma=g)
    assert r1["visits"] == {"L": 1, "R": 0}
    assert r1["Q"]["L"] == 1.0
    r2 = muzero(None, ["L", "R"], _rep, _dyn, _pred, simulations=2, gamma=g)
    assert r2["Q"]["L"] == pytest.approx(1.0 + g * 0.5, rel=1e-15)
    r3 = muzero(None, ["L", "R"], _rep, _dyn, _pred, simulations=12, temperature=1.0)
    tot = sum(r3["visits"].values())
    assert list(r3["policy"]) == pytest.approx([r3["visits"]["L"] / tot, r3["visits"]["R"] / tot], rel=1e-15)


