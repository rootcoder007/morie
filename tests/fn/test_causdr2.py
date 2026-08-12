"""Tests for causdr2 (DML2 partially linear model).

Replaces the generated stub, which imported ``causal_dr_orthogonal``.
"""

import math

from morie.fn.causdr2 import causdr2


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _gauss(r):
    return math.sqrt(-2 * math.log(max(r(), 1e-12))) * \
        math.cos(2 * math.pi * r())


def _panel(theta=2.0, n=400, seed=3, confound=True):
    r = _lcg(seed)
    y, d, X = [], [], []
    for _ in range(n):
        x1, x2 = _gauss(r), _gauss(r)
        g = 1.5 * x1 - 0.8 * x2 if confound else 0.0
        di = (0.9 * x1 + 0.4 * x2 if confound else 0.0) + _gauss(r)
        y.append(theta * di + g + 0.5 * _gauss(r))
        d.append(di)
        X.append([x1, x2])   # no intercept: the module adds its own
    return y, d, X


def test_the_treatment_effect_is_recovered_under_confounding():
    y, d, X = _panel(theta=2.0)
    res = causdr2(y, d, X, K=2, seed=1)
    assert abs(res["estimate"] - 2.0) < 0.15
    assert res["se"] > 0


def test_a_naive_regression_of_y_on_d_would_be_biased():
    # the point of partialling out: without it the confounder inflates
    # the slope
    y, d, X = _panel(theta=2.0)
    n = len(y)
    dbar = sum(d) / n
    ybar = sum(y) / n
    naive = (sum((d[i] - dbar) * (y[i] - ybar) for i in range(n)) /
             sum((d[i] - dbar) ** 2 for i in range(n)))
    res = causdr2(y, d, X, K=2, seed=1)
    assert abs(naive - 2.0) > abs(res["estimate"] - 2.0)


def test_no_confounding_still_recovers_the_effect():
    y, d, X = _panel(theta=1.0, confound=False)
    res = causdr2(y, d, X, K=2, seed=1)
    assert abs(res["estimate"] - 1.0) < 0.15


def test_more_folds_still_agree():
    y, d, X = _panel(theta=2.0)
    a = causdr2(y, d, X, K=2, seed=1)["estimate"]
    b = causdr2(y, d, X, K=5, seed=1)["estimate"]
    assert abs(a - b) < 0.2
    # "folds" is the per-observation assignment, not the count
    folds = causdr2(y, d, X, K=5, seed=1)["folds"]
    assert len(folds) == len(y)
    assert sorted(set(folds)) == [1, 2, 3, 4, 5]


def test_the_standard_error_shrinks_with_the_sample():
    small = causdr2(*_panel(n=120, seed=7), K=2, seed=1)["se"]
    large = causdr2(*_panel(n=800, seed=7), K=2, seed=1)["se"]
    assert large < small


def test_validation():
    y, d, X = _panel(n=60)
    for call in (lambda: causdr2(y[:-1], d, X),
                 lambda: causdr2(y, d, X, K=0),
                 lambda: causdr2(y, [1.0] * len(y), X),
                 # an intercept column is the easy mistake, and it used
                 # to surface as a bare "singular matrix"
                 lambda: causdr2(y, d, [[1.0] + row for row in X])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
