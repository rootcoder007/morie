import math

import pytest

from morie.fn.symreg import pysr_regression

X = [[i / 4.0, math.cos(i / 3.0)] for i in range(-10, 11)]
Y = [2.5 * r[0] * r[0] - r[1] + 0.3 for r in X]


def test_front_is_pareto_and_scores_recompute():
    r = pysr_regression(X, Y, niterations=15, population_size=40, seed=3)
    eq = r.equations
    for a, b in zip(eq, eq[1:]):
        assert b["complexity"] > a["complexity"] and b["loss"] < a["loss"]
        want = -(math.log(max(b["loss"], 1e-300)) - math.log(max(a["loss"], 1e-300))) / (
            b["complexity"] - a["complexity"]
        )
        assert b["score"] == pytest.approx(want, abs=1e-12)
    mse = sum((p - v) ** 2 for p, v in zip(r.prediction, Y)) / len(Y)
    assert r.best["loss"] == pytest.approx(mse, rel=1e-12, abs=1e-15)
    lmin = min(e["loss"] for e in eq)
    assert r.best["loss"] <= 1.5 * lmin


def test_exact_recovery_and_determinism():
    X1 = [[i / 4.0] for i in range(-8, 9)]
    y1 = [2.0 * r[0] * r[0] + r[0] for r in X1]
    a = pysr_regression(X1, y1, niterations=40, seed=3)
    b = pysr_regression(X1, y1, niterations=40, seed=3)
    assert a.best == b.best
    assert a.best["loss"] < 1e-20


def test_unary_operators_and_errors():
    r = pysr_regression(X, Y, unary_operators=("cos", "exp"), niterations=5, population_size=30, seed=5)
    assert r.equations and all(math.isfinite(e["loss"]) for e in r.equations)
    with pytest.raises(ValueError):
        pysr_regression(X, Y, unary_operators=("tanh",))
    with pytest.raises(ValueError):
        pysr_regression(X, Y[:-1])
