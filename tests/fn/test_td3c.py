"""Tests for td3c.td3: argument mapping onto geron_td3."""

from morie.fn.hmtd3 import geron_td3
from morie.fn.td3c import td3


class Chain:
    n_states, n_actions = 3, 2

    def __init__(self):
        self.s = 0

    def reset(self):
        self.s = 0
        return 0

    def step(self, a):
        self.s = min(2, self.s + 1) if a == 1 else max(0, self.s - 1)
        return self.s, float(self.s == 2), self.s == 2


def test_same_as_geron_td3():
    Q = [[0.0, 0.1], [0.0, 0.1], [0.0, 0.0]]
    a = td3(Chain(), actor=[1, 1, 0], critic1=Q, critic2=Q, epochs=25, seed=3)
    b = geron_td3(Chain(), policy=[1, 1, 0], Q1=Q, Q2=Q, epochs=25, seed=3)
    assert list(a["policy"]) == list(b["policy"])
    assert a["overestimation_gap"] == b["overestimation_gap"]
