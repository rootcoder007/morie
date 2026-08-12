"""Tests for pate (Papernot et al. 2017, PATE)."""

import math

from morie.fn import _array_core as np

from morie.fn.pate import (epsilon_data_independent, lemma4_bound,
                           moments_accountant, noisy_argmax, pate,
                           pate_aggregate, private_aggregation,
                           teacher_votes, theorem3_moment)


def test_printed_epsilon_values():
    """The paper's own worked numbers, section 3.2."""
    assert 25.5 < epsilon_data_independent(1000, 0.05, 1e-6) < 27.0
    assert round(epsilon_data_independent(100, 0.05, 1e-5), 2) == 5.80


def test_epsilon_formula():
    for T, g, d in ((1000, 0.05, 1e-6), (37, 0.2, 1e-3)):
        want = 4.0 * T * g ** 2 + 2.0 * g * math.sqrt(2.0 * T *
                                                      math.log(1.0 / d))
        assert abs(epsilon_data_independent(T, g, d) - want) < 1e-12
    assert epsilon_data_independent(0, 0.05, 1e-5) == 0.0


def test_lemma4_bounds_the_miss_probability():
    rng = np.random.default_rng(4)
    for counts, gamma in (([100, 2, 1], 0.05), ([30, 28, 25], 0.2)):
        js = max(range(len(counts)), key=lambda t: counts[t])
        n = 2000
        miss = sum(1 for _ in range(n)
                   if noisy_argmax(counts, gamma, rng) != js)
        assert miss / float(n) <= lemma4_bound(counts, gamma)[0] + 0.03


def test_noisy_argmax_limits():
    rng = np.random.default_rng(1)
    assert set(noisy_argmax([10, 8, 1], 1e6, rng) for _ in range(100)) == {0}
    draws = [noisy_argmax([10, 8, 1], 1e-3, rng) for _ in range(2000)]
    freq = [draws.count(j) / 2000.0 for j in range(3)]
    assert all(abs(v - 1 / 3.0) < 0.06 for v in freq)


def test_theorem3_condition_and_formula():
    gamma = 0.05
    limit = (math.exp(2 * gamma) - 1) / (math.exp(4 * gamma) - 1)
    assert theorem3_moment(limit, gamma, 3) is None
    q, l = 0.01, 4
    want = math.log((1 - q) * ((1 - q) / (1 - math.exp(2 * gamma) * q)) ** l
                    + q * math.exp(2 * gamma * l))
    assert abs(theorem3_moment(q, gamma, l) - want) < 1e-12
    assert theorem3_moment(0.0, gamma, 5) == 0.0


def test_accountant_composes_and_inverts_the_tail_bound():
    one = moments_accountant([[100, 2, 1]], 0.05, 1e-5)
    ten = moments_accountant([[100, 2, 1]] * 10, 0.05, 1e-5)
    for l in one["alpha"]:
        assert abs(ten["alpha"][l] - 10.0 * one["alpha"][l]) < 1e-12
    lam = ten["lambda"]
    assert abs(ten["epsilon"] -
               (ten["alpha"][lam] + math.log(1e5)) / lam) < 1e-12
    assert abs(math.exp(ten["alpha"][lam] - lam * ten["epsilon"]) -
               1e-5) < 1e-12


def test_strong_quorum_is_cheaper_than_a_split_vote():
    strong = moments_accountant([[100, 2, 1]] * 100, 0.05, 1e-5)
    weak = moments_accountant([[34, 33, 33]] * 100, 0.05, 1e-5)
    assert strong["epsilon"] < weak["epsilon"]
    assert strong["used"]["data_dependent"] == 100
    assert weak["used"]["data_independent"] == 100


def _teachers(n=25, flip=1):
    def make(bias):
        def predict(rows):
            out = []
            for i, x in enumerate(rows):
                true = 1 if sum(x) >= 2 else 0
                out.append(1 - true if (i * 7 + bias) % 10 < flip else true)
            return out
        return predict
    return [make(b) for b in range(n)]


def test_end_to_end():
    rows = [[(i >> b) & 1 for b in range(3)] for i in range(8)] * 5
    res = pate(_teachers(), rows, gamma=0.05, delta=1e-5, n_classes=2)
    assert len(res["labels"]) == len(rows)
    assert all(sum(v) == 25 for v in res["votes"])
    assert 0.5 < res["agreement"] < 1.0
    assert res["epsilon"] <= res["epsilon_data_independent"] + 1e-12
    assert res["epsilon"] <= res["epsilon_accountant"] + 1e-12
    louder = pate(_teachers(), rows, gamma=1.0, delta=1e-5, n_classes=2)
    assert louder["agreement"] >= res["agreement"]


def test_validation():
    rows = [[1, 0, 1]]
    for call in (lambda: noisy_argmax([1, 2], 0.0),
                 lambda: epsilon_data_independent(10, 0.05, 1.0),
                 lambda: teacher_votes([], rows),
                 lambda: pate(_teachers(), []),
                 lambda: lemma4_bound([], 0.1),
                 lambda: moments_accountant([[5, 1]], 0.05, 1e-5,
                                            lambdas=[0])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_aliases():
    assert private_aggregation is pate and pate_aggregate is pate
