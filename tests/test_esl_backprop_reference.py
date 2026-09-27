"""Back-propagation (ESL eqs 11.13-11.15) against central finite differences; weight decay (eq 11.16)."""

import copy
import math

from morie.fn import esl_backprop, esl_weight_decay


def weights():
    return {
        "alpha": [[0.3, -0.2, 0.5], [0.1, 0.4, -0.3]],
        "alpha0": [0.05, -0.1, 0.2],
        "beta": [[0.7, -0.4], [-0.5, 0.3], [0.2, 0.6]],
        "beta0": [0.1, -0.2],
    }


def data():
    X = [[math.sin(t), math.cos(2 * t)] for t in range(1, 21)]
    return X


def fd_check(X, y, task):
    w = weights()
    g = esl_backprop(X, y, w, task=task)
    h = 1e-6
    for key, gkey in (
        ("alpha", "grad_alpha"),
        ("alpha0", "grad_alpha0"),
        ("beta", "grad_beta"),
        ("beta0", "grad_beta0"),
    ):
        G = g[gkey].tolist()
        flat = (
            [(i, None) for i in range(len(w[key]))]
            if not isinstance(w[key][0], list)
            else [(i, j) for i in range(len(w[key])) for j in range(len(w[key][0]))]
        )
        for i, j in flat:
            up, dn = copy.deepcopy(w), copy.deepcopy(w)
            if j is None:
                up[key][i] += h
                dn[key][i] -= h
                an = G[i]
            else:
                up[key][i][j] += h
                dn[key][i][j] -= h
                an = G[i][j]
            num = (esl_backprop(X, y, up, task=task)["loss"] - esl_backprop(X, y, dn, task=task)["loss"]) / (2 * h)
            assert abs(num - an) < 1e-7, (key, i, j, num, an)


def test_regression_gradients_equal_finite_differences():
    X = data()
    y = [[math.sin(3 * t), math.cos(t)] for t in range(1, 21)]
    fd_check(X, y, "regression")


def test_classification_gradients_equal_finite_differences():
    X = data()
    fd_check(X, [t % 2 for t in range(1, 21)], "classification")


def test_weight_decay_penalty():
    r = esl_weight_decay([0.3, -0.2, 0.5, 0.7], lambda_=0.1, loss=2.0)
    assert abs(r["penalty"] - 0.1 * (0.09 + 0.04 + 0.25 + 0.49)) < 1e-15  # ESL eq 11.16: lambda sum w^2
    assert [round(float(v), 15) for v in r["gradient"]] == [0.06, -0.04, 0.1, 0.14]
    assert abs(r["objective"] - (2.0 + r["penalty"])) < 1e-15
