"""Tests for bcq.bcq."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.bcq import bcq


def test_bcq_basic():
    """Test basic functionality."""
    # Build a small tabular dataset of (s, a, r, s_next[, done]) tuples.
    # The docstring specifies transitions of arity 4 or 5 with hashable
    # state/action labels.
    rng = np.random.default_rng(42)
    states = ["s0", "s1", "s2"]
    actions = ["a0", "a1"]
    dataset = []
    for _ in range(20):
        s = states[int(rng.integers(0, len(states)))]
        a = actions[int(rng.integers(0, len(actions)))]
        s_next = states[int(rng.integers(0, len(states)))]
        r = float(rng.normal(0, 1))
        done = bool(rng.integers(0, 2))
        dataset.append((s, a, r, s_next, done))

    result = bcq(dataset, states=states, actions=actions,
                 tau=0.3, gamma=0.99, lr=0.5, iters=200,
                 loss="huber", huber_c=1.0)

    # Documented return: a mapping with both 'estimate' and 'q' keys for Q.
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "q" in result
    assert "policy" in result
    assert "allowed" in result
    assert "behavior" in result
    assert "value" in result
    assert "n_eliminated" in result
    assert "bellman_error" in result

    # Q is {(s, a): value} and must cover every (state, action) pair.
    Q = result["estimate"]
    assert set(Q.keys()) == {(s, a) for s in states for a in actions}
    for v in Q.values():
        assert isinstance(v, float)

    # Allowed actions per state are a non-empty subset of the action set.
    allowed = result["allowed"]
    for s in states:
        assert set(allowed[s]).issubset(set(actions))
        assert len(allowed[s]) >= 1

    # tau = 0 keeps every action (constraint set is the whole action set).
    assert all(set(allowed[s]) == set(actions) for s in states)

    # The constrained argmax policy picks an action from allowed[s].
    policy = result["policy"]
    for s in states:
        assert policy[s] in allowed[s]

    # State values are the constrained max of Q over allowed[s], computed
    # here independently from the returned Q and allowed maps.
    value = result["value"]
    for s in states:
        expected_v = max(Q[(s, a)] for a in allowed[s])
        assert value[s] == expected_v

    # n_eliminated is the total number of removed (s, a) pairs.
    expected_elim = sum(len(actions) - len(allowed[s]) for s in states)
    assert result["n_eliminated"] == expected_elim

    # Behavior policy G is a probability distribution over actions per
    # state (since the default is the empirical batch frequencies).
    behavior = result["behavior"]
    for s in states:
        total = sum(behavior[(s, a)] for a in actions)
        assert abs(total - 1.0) < 1e-9
        for a in actions:
            assert 0.0 <= behavior[(s, a)] <= 1.0

    # Bellman error is a non-negative scalar.
    be = result["bellman_error"]
    assert isinstance(be, float)
    assert be >= 0.0


def test_bcq_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    states = ["s0", "s1"]
    actions = ["a0", "a1", "a2"]
    dataset = []
    for _ in range(15):
        s = states[int(rng.integers(0, len(states)))]
        a = actions[int(rng.integers(0, len(actions)))]
        s_next = states[int(rng.integers(0, len(states)))]
        r = float(rng.normal(0, 1))
        dataset.append((s, a, r, s_next))

    # tau = 1 with a non-degenerate behavior falls back to imitation of G
    # (the paper's stated behavior), so the policy must still be defined
    # and allowed must be non-empty.
    result = bcq(dataset, states=states, actions=actions,
                 tau=1.0, gamma=0.9, lr=0.5, iters=100,
                 loss="squared")

    assert isinstance(result, dict)
    for s in states:
        assert len(result["allowed"][s]) >= 1
        assert result["policy"][s] in result["allowed"][s]

    # Bellman error must be finite and non-negative under squared loss.
    be = result["bellman_error"]
    assert isinstance(be, float)
    assert be >= 0.0
