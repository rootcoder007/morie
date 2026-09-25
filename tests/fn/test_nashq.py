"""Tests for nashq.nash_q_learning (Hu & Wellman 2003, Table 2)."""

import pytest

from morie.fn.nashq import nash_q_learning

A = (["C", "D"], ["C", "D"])
PD = {("C", "C"): (3.0, 3.0), ("C", "D"): (0.0, 5.0), ("D", "C"): (5.0, 0.0), ("D", "D"): (1.0, 1.0)}


def test_nashq_basic():
    """A one-shot prisoner's dilemma with uniform exploration: every
    joint action is revisited until Q = (1 - 2^-k) r reaches the stage
    payoffs exactly, and the Nash policy is mutual defection."""
    r = nash_q_learning(["s", "end"], A, lambda s, a, b: "end",
                        lambda s, a, b, s2: PD[(a, b)], epsilon=1.0,
                        episodes=300, terminal=("end",), alpha=0.5)
    assert isinstance(r, dict)
    assert r["q"][(0, "s")] == [[3.0, 0.0], [5.0, 1.0]]
    assert r["q"][(1, "s")] == [[3.0, 5.0], [0.0, 1.0]]
    assert r["policy"]["s"] == ([0.0, 1.0], [0.0, 1.0])


def test_nashq_edge():
    """Two stages, s0 then the dilemma at s1: the update target is
    r0 + gamma * NashValue(s1), and the dilemma's Nash value is (1, 1),
    so Q(s0) = r0 + 0.9 entrywise."""
    R0 = {("C", "C"): (2.0, 1.0), ("C", "D"): (0.0, 0.0), ("D", "C"): (0.0, 0.0), ("D", "D"): (1.0, 2.0)}

    def step(s, a, b):
        return "s1" if s == "s0" else "end"

    def rew(s, a, b, s2):
        return R0[(a, b)] if s == "s0" else PD[(a, b)]

    r = nash_q_learning(["s0", "s1", "end"], A, step, rew, gamma=0.9, epsilon=1.0,
                        episodes=600, terminal=("end",), alpha=0.5)
    for p in (0, 1):
        exp = [[R0[(a, b)][p] + 0.9 * 1.0 for b in A[1]] for a in A[0]]
        for row, erow in zip(r["q"][(p, "s0")], exp):
            assert row == pytest.approx(erow, rel=1e-12)


