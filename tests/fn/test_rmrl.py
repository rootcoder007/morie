"""Tests for rmrl.reward_machine (Toro Icarte et al. 2018, Defs. 3.1-3.2)."""

import pytest

from morie.fn.rmrl import reward_machine, reward_machine_run

# "get coffee, then deliver it to the office without stepping on a
# decoration": u0 -c-> u1, u1 -o-> u2 (reward 1), decorations fail
EDGES = [
    (0, (["c"], ["d"]), 1, 0.0),
    (0, (["d"], []), 3, 0.0),
    (1, (["o"], ["d"]), 2, 1.0),
    (1, (["d"], []), 3, 0.0),
]


def test_rmrl_basic():
    """delta_u and delta_r follow the first matching edge; with no match
    the machine stays and pays 0 (the implicit <true, 0> self-loop)."""
    rm = reward_machine(EDGES, u0=0, terminal=(2, 3))
    assert rm.step(0, {"c"}) == (1, 0.0)
    assert rm.step(0, set()) == (0, 0.0)
    assert rm.step(0, {"c", "d"}) == (3, 0.0)     # negative literal blocks c
    assert rm.step(1, {"o"}) == (2, 1.0)
    assert rm.step(2, {"c"}) == (2, 0.0)          # terminal absorbs
    assert rm.states == {0, 1, 2, 3}


def test_rmrl_edge():
    """A labelled trace accumulates the edge rewards and stops in the
    terminal state; an edge that is not a 4-tuple is refused."""
    rm = reward_machine(EDGES, u0=0, terminal=(2, 3))
    out = reward_machine_run(rm, [set(), {"c"}, set(), {"o"}, {"c"}])
    assert out["states"] == [0, 0, 1, 1, 2, 2]
    assert out["rewards"] == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert out["total_reward"] == 1.0 and out["final_state"] == 2 and out["accepted"]
    with pytest.raises(ValueError):
        reward_machine([(0, "true", 1)])
