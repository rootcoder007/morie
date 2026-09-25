"""Tests for mtdrl.meta_rl (Wang et al. 2016, sec. 2 evaluation loop)."""

import pytest

from morie.fn.mtdrl import bandit_tasks, history_features, meta_rl


class Always0:
    """Frozen agent that always pulls arm 0 and counts its resets."""

    def __init__(self):
        self.resets = 0
        self.seen = []

    def reset(self):
        self.resets += 1

    def act(self, features, rng):
        self.seen.append(features)
        return 0

    def observe(self, action, reward):
        pass


def test_mtdrl_basic():
    """Deterministic arms: task [1, 0] pays every step, task [0, 1]
    never does and costs regret 1 per step; the agent is reset once per
    episode and sees (one-hot previous action, previous reward, step)."""
    ag = Always0()
    r = meta_rl([[1.0, 0.0], [0.0, 1.0]], ag, episode_length=5)
    assert r["episode_reward"] == [5.0, 0.0]
    assert r["mean_reward"] == pytest.approx(0.5, abs=1e-15)
    assert r["regret"] == pytest.approx(5.0, abs=1e-15)
    assert r["reward_by_step"] == [0.5] * 5
    assert r["optimal_action_rate"] == [0.5] * 5
    assert ag.resets == 2
    assert ag.seen[0] == [0.0, 0.0, 0.0, 0.0] and ag.seen[1] == [1.0, 0.0, 1.0, 1.0]
    assert history_features([(1, 0.0)], 3) == [0.0, 1.0, 0.0, 0.0, 1.0]
    ag2 = Always0()
    meta_rl([[1.0, 0.0]] * 3, ag2, episode_length=2, reset_between_episodes=False)
    assert ag2.resets == 0
    assert all(abs(a + b - 1.0) < 1e-15 for a, b in bandit_tasks(2, 20, structure="paired"))


def test_mtdrl_edge():
    """Ragged tasks, an agent lacking the interface, and an empty task
    list raise."""
    with pytest.raises(ValueError):
        meta_rl([[0.5, 0.5], [0.5]], Always0())
    with pytest.raises(TypeError):
        meta_rl([[0.5, 0.5]], object())
    with pytest.raises(ValueError):
        meta_rl([], Always0())
