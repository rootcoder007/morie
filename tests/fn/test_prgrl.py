"""Tests for prgrl (Bengio et al. 2009, curriculum learning)."""

import math

from morie.fn.prgrl import (curriculum_schedule, entropy, is_curriculum,
                            prgrl)


def test_the_schedule_satisfies_the_papers_definition():
    lam, w, q = curriculum_schedule([5, 1, 4, 2, 3], n_steps=5)
    assert abs(lam[-1] - 1.0) < 1e-12
    assert all(abs(v - 1.0) < 1e-12 for v in w[-1])       # eqn 2
    assert all(abs(v - 0.2) < 1e-12 for v in q[-1])
    chk = is_curriculum(w)
    assert chk["is_curriculum"]
    assert chk["strictly_increasing"] and chk["weights_monotone"]
    assert chk["final_step_is_p"]


def test_entropy_of_the_uniform_steps_is_log_k():
    _, _, q = curriculum_schedule([5, 1, 4, 2, 3], n_steps=5)
    for k in range(1, 6):
        assert abs(entropy(q[k - 1]) - math.log(k)) < 1e-12


def test_the_easiest_example_is_added_first():
    _, w, _ = curriculum_schedule([5, 1, 4, 2, 3], n_steps=5)
    assert w[0][1] == 1.0 and sum(w[0]) == 1.0
    _, hw, _ = curriculum_schedule([5, 1, 4, 2, 3], n_steps=5,
                                   hard_first=True)
    assert hw[0][0] == 1.0


def test_dropping_an_example_breaks_the_monotonicity_condition():
    bad = [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]
    res = is_curriculum(bad)
    assert res["entropy_increasing"]          # eqn 3 can still hold
    assert not res["weights_monotone"]        # but eqn 4 fails
    assert not res["is_curriculum"]


def test_a_sequence_that_never_grows_is_not_strictly_a_curriculum():
    same = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert not is_curriculum(same)["strictly_increasing"]


def _panel():
    st = [7]

    def r():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)

    def g():
        return math.sqrt(-2 * math.log(max(r(), 1e-12))) * \
            math.cos(2 * math.pi * r())

    X, y, d = [], [], []
    for i in range(80):
        hard = i >= 40
        x1, x2 = g(), g()
        X.append([x1, x2])
        y.append(2.0 * x1 - 1.0 * x2 + (2.0 if hard else 0.02) * g())
        d.append(1.0 if hard else 0.0)
    return X, y, d


def test_on_a_convex_objective_the_ordering_changes_nothing():
    X, y, d = _panel()
    res = prgrl(X, y, d, n_steps=4, epochs_per_step=60, lr=0.1)
    assert res["is_curriculum"]
    assert abs(res["curriculum_loss"] - res["baseline_loss"]) < 1e-7
    for k in range(2):
        assert abs(res["curriculum_beta"][k] -
                   res["baseline_beta"][k]) < 1e-4


def test_both_routes_recover_the_planted_coefficients():
    X, y, d = _panel()
    res = prgrl(X, y, d, n_steps=4, epochs_per_step=60, lr=0.1)
    assert abs(res["curriculum_beta"][0] - 2.0) < 0.2
    assert abs(res["curriculum_beta"][1] + 1.0) < 0.2
    assert len(res["curriculum_history"]) == \
        len(res["baseline_history"]) == 240
    assert res["curriculum_history"][-1] < res["curriculum_history"][0]


def test_validation():
    for call in (lambda: curriculum_schedule([1.0]),
                 lambda: curriculum_schedule([1.0, 2.0], n_steps=1),
                 lambda: is_curriculum([[1.0, 1.0]]),
                 lambda: is_curriculum([[1.0, 1.0], [1.0]]),
                 lambda: is_curriculum([[2.0, 1.0], [1.0, 1.0]]),
                 lambda: entropy([0.0, 0.0]),
                 lambda: prgrl([[1.0]], [1.0], [1.0]),
                 lambda: prgrl([[1.0], [2.0]], [1.0], [1.0, 2.0]),
                 lambda: prgrl([[1.0], [2.0]], [1.0, 2.0], [1.0, 2.0],
                               lr=0.0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
