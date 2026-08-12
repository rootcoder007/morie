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


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _gauss(r):
    return math.sqrt(-2 * math.log(max(r(), 1e-12))) * \
        math.cos(2 * math.pi * r())


_MEAN = 1.0 / math.sqrt(2.0)


def _gaussians(r, n):
    X, y, d = [], [], []
    for _ in range(n):
        c = 1.0 if r() < 0.5 else -1.0
        x = [c * _MEAN + _gauss(r), c * _MEAN + _gauss(r)]
        X.append(x)
        y.append(c)
        d.append(-(c * (x[0] + x[1])))
    return X, y, d


def test_section_4_1_reproduces():
    from morie.fn.prgrl import easy_only_fit
    r = _lcg(5)
    Xtr, ytr, dtr = _gaussians(r, 50)
    Xte, yte, _ = _gaussians(r, 20000)
    res = easy_only_fit(Xtr, ytr, dtr, Xte, yte, quantile=0.7,
                        updates=200, n_repeats=100, seed=2)
    assert res["improvement"] > 0
    assert abs(res["easy_only_error"] - 0.163) < 0.02   # paper's number


def _perceptron_panel(r, w_true, n, p_rel=3, p_irr=7):
    X, y, d = [], [], []
    for _ in range(n):
        xr = [r() for _ in range(p_rel)]
        nz = int(r() * (p_irr + 1))
        xi = [(r() if k < nz else 0.0) for k in range(p_irr)]
        s = sum(w_true[k] * xr[k] for k in range(p_rel))
        X.append(xr + xi)
        y.append(1.0 if s > 0 else -1.0)
        d.append(float(nz))
    return X, y, d


def test_section_4_2_runs_both_orderings_and_is_measured_held_out():
    r = _lcg(11)
    w_true = [_gauss(r) for _ in range(3)]
    Xp, yp, dp = _perceptron_panel(r, w_true, 200)
    Xq, yq, _ = _perceptron_panel(r, w_true, 2000)
    got = {}
    for o in ("sorted", "sampled"):
        res = prgrl(Xp, yp, dp, X_test=Xq, y_test=yq, updates=200,
                    n_repeats=100, seed=3, order=o)
        assert res["held_out"] and res["order"] == o
        assert 0.0 <= res["curriculum_error"] <= 1.0
        assert res["is_curriculum"]
        got[o] = res
    # sampling from Q_lambda beats sorting once, as eqns 1-2 imply
    assert got["sampled"]["curriculum_error"] < \
        got["sorted"]["curriculum_error"]


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
                               updates=0),
                 lambda: prgrl([[1.0], [2.0]], [0.0, 1.0], [1.0, 2.0]),
                 lambda: prgrl([[1.0], [2.0]], [1.0, -1.0], [1.0, 2.0],
                               order="reverse")):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
