"""Tests for meta1l (S-, T-, X- and R-metalearners for the CATE).

Replaces the generated stub, which imported ``meta_learner_ensemble``.
"""

import math

from morie.fn.meta1l import meta1l


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _gauss(r):
    return math.sqrt(-2 * math.log(max(r(), 1e-12))) * \
        math.cos(2 * math.pi * r())


def _panel(n=400, seed=5, tau_slope=2.0, base=1.0):
    """CATE is base + tau_slope * x1, so the truth is known per unit."""
    r = _lcg(seed)
    y, w, X = [], [], []
    for i in range(n):
        x1, x2 = _gauss(r), _gauss(r)
        wi = 1 if r() < 0.5 else 0
        tau = base + tau_slope * x1
        y.append(0.5 * x1 - 0.3 * x2 + wi * tau + 0.2 * _gauss(r))
        w.append(wi)
        X.append([x1, x2])
    return y, w, X


def _truth(X, tau_slope=2.0, base=1.0):
    return [base + tau_slope * row[0] for row in X]


def test_the_three_learners_that_can_express_it_recover_a_linear_cate():
    # T, X and R fit the arms separately (or orthogonalise), so they can
    # follow a CATE that varies with x1. The S-learner cannot -- see the
    # next test -- which is why it is not in this list.
    y, w, X = _panel()
    res = meta1l(y, w, X)
    truth = _truth(X)
    for key in ("cate_t", "cate_x", "cate_r"):
        est = res[key]
        err = max(abs(est[i] - truth[i]) for i in range(len(truth)))
        assert err < 0.6, key


def test_the_s_learner_misses_a_heterogeneous_effect_it_cannot_express():
    # the S-learner puts w in as one more regressor with no interaction,
    # so it can only ever report a constant effect
    y, w, X = _panel()
    res = meta1l(y, w, X)
    truth = _truth(X)
    spread = max(res["cate_s"]) - min(res["cate_s"])
    assert spread < 1e-6                       # a single number for all
    err_s = max(abs(res["cate_s"][i] - truth[i])
                for i in range(len(truth)))
    err_t = max(abs(res["cate_t"][i] - truth[i])
                for i in range(len(truth)))
    assert err_s > 5.0 > err_t                 # measured: 5.9 against 0.15
    assert max(res["cate_t"]) - min(res["cate_t"]) > 1.0


def test_a_constant_effect_is_recovered_by_every_learner():
    y, w, X = _panel(tau_slope=0.0, base=3.0)
    res = meta1l(y, w, X)
    for key in ("cate_s", "cate_t", "cate_x", "cate_r"):
        mean = sum(res[key]) / len(res[key])
        assert abs(mean - 3.0) < 0.3, key


def test_the_reported_average_effect_matches_its_learner():
    y, w, X = _panel()
    res = meta1l(y, w, X)
    # "estimate" carries one average effect per learner
    assert sorted(res["estimate"]) == ["r", "s", "t", "x"]
    assert abs(res["estimate"]["t"] -
               sum(res["cate_t"]) / len(res["cate_t"])) < 0.5
    assert res["n"] == len(y)
    assert res["n_treat"] == sum(w)


def test_validation():
    y, w, X = _panel(n=60)
    for call in (lambda: meta1l(y[:-1], w, X),
                 lambda: meta1l(y, [2] * len(y), X),
                 lambda: meta1l(y[:6], w[:6], X[:6])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
