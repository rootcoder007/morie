"""Tests for ucbb.ucb_bandit.

Anchor: the play sequence of the deterministic UCB1 policy (Auer,
Cesa-Bianchi, Fischer 2002, figure 1) recomputed by independent
arithmetic in the test, index by index.
"""

import math

from morie.fn.ucbb import ucbb


def test_ucbb_hand_played_sequence():
    # K = 2 with constant rewards 1.0 and 0.0. After the forced first
    # two plays, arm 0 has index 1 + sqrt(2 ln n) >= arm 1 index
    # sqrt(2 ln n) for every n, so arm 0 is played from then on.
    x = [[1.0, 0.0]] * 6
    r = ucbb(x)
    assert list(r["actions"]) == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert r["counts"][0] == 5.0 and r["counts"][1] == 1.0
    assert r["estimate"] == 0.0
    assert abs(r["total_reward"] - 5.0) < 1e-15


def test_ucbb_indices_recomputed_independently():
    # replay the policy by independent arithmetic and compare play by play
    x = [
        [0.3, 0.9, 0.5],
        [0.2, 0.1, 0.8],
        [0.9, 0.4, 0.1],
        [0.7, 0.6, 0.2],
        [0.1, 0.8, 0.3],
        [0.5, 0.5, 0.5],
        [0.4, 0.2, 0.9],
    ]
    K = 3
    counts = [0] * K
    sums = [0.0] * K
    expect = []
    for t in range(len(x)):
        if t < K:
            j = t
        else:
            n = t
            idx = [sums[k] / counts[k] + math.sqrt(2.0 * math.log(n) / counts[k])
                   for k in range(K)]
            j = max(range(K), key=lambda k: (idx[k], -k))
            # ties to lowest index: max over (value, -k)
        counts[j] += 1
        sums[j] += x[t][j]
        expect.append(float(j))
    r = ucbb(x)
    assert list(r["actions"]) == expect
    for k in range(K):
        assert abs(r["means"][k] - sums[k] / counts[k]) < 1e-15


def test_ucbb_requires_enough_plays():
    try:
        ucbb([[0.5, 0.5]], T=1)
    except ValueError:
        return
    raise AssertionError("T < K accepted")
