"""Tests for acrt.actor_critic."""

from morie.fn import _array_core as np
from morie.fn.acrt import actor_critic


def test_acrt_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    result = actor_critic(env, actor, critic)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_acrt_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    result = actor_critic(env, actor, critic)
    assert isinstance(result, dict)


def test_acrt_updates_recomputed():
    """One-step actor-critic (Sutton-Barto eqs 13.12-13.14) by hand."""
    import pytest

    R = [1.0, 0.0, 2.0]
    V = [0.5, 0.2, 0.1, 0.0]
    G = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    g, at, aw = 0.9, 0.1, 0.2
    th, w, i_fac, deltas = [0.0, 0.0], [0.0], 1.0, []
    for t in range(3):
        d = R[t] + g * V[t + 1] - V[t]
        deltas.append(d)
        th = [th[j] + at * i_fac * d * G[t][j] for j in range(2)]
        w = [w[0] + aw * d]
        i_fac *= g
    r = actor_critic(None, rewards=R, values=V, grad_logpi=G, alpha_theta=at, alpha_w=aw, gamma=g)
    assert r["deltas"] == pytest.approx(deltas, rel=1e-14)
    assert r["theta"] == pytest.approx(th, rel=1e-14)
    assert r["w"] == pytest.approx(w, rel=1e-14)
    assert r["estimate"] == pytest.approx(sum(deltas) / 3, rel=1e-14)
