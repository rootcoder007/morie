"""Tests for lbfgsm (limited-memory BFGS).

Replaces the generated stub, which imported ``lbfgs``.
"""

import math

from morie.fn.lbfgsm import lbfgs_minimize


def test_a_quadratic_is_solved_to_its_exact_minimum():
    # f(x) = (x0 - 3)^2 + 2 (x1 + 1)^2, minimum at (3, -1)
    def f(x):
        return (x[0] - 3.0) ** 2 + 2.0 * (x[1] + 1.0) ** 2

    def g(x):
        return [2.0 * (x[0] - 3.0), 4.0 * (x[1] + 1.0)]

    res = lbfgs_minimize(f, [0.0, 0.0], g)
    assert res["converged"]
    assert abs(res["x"][0] - 3.0) < 1e-6
    assert abs(res["x"][1] + 1.0) < 1e-6
    assert res["fun"] < 1e-12


def test_rosenbrock_is_solved():
    def f(x):
        return (1 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    def g(x):
        return [-2 * (1 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
                200.0 * (x[1] - x[0] ** 2)]

    res = lbfgs_minimize(f, [-1.2, 1.0], g, max_iter=500)
    assert abs(res["x"][0] - 1.0) < 1e-4
    assert abs(res["x"][1] - 1.0) < 1e-4


def test_the_gradient_norm_falls_below_the_tolerance():
    def f(x):
        return sum(v * v for v in x)

    def g(x):
        return [2.0 * v for v in x]

    res = lbfgs_minimize(f, [1.0, 2.0, 3.0], g, tol=1e-10)
    assert res["grad_norm"] <= 1e-10
    assert res["converged"]


def test_starting_at_the_minimum_stops_immediately():
    res = lbfgs_minimize(lambda x: x[0] ** 2, [0.0],
                         lambda x: [2.0 * x[0]])
    # it checks the gradient at the start, so it stops after a single
    # pass rather than doing no work at all
    assert res["iterations"] <= 1
    assert res["converged"]
    assert abs(res["x"][0]) < 1e-12


def test_the_history_is_monotone_for_a_convex_problem():
    def f(x):
        return math.cosh(x[0]) + x[0] ** 2

    def g(x):
        return [math.sinh(x[0]) + 2.0 * x[0]]

    res = lbfgs_minimize(f, [3.0], g)
    hist = res["history"]
    assert all(hist[i + 1] <= hist[i] + 1e-12
               for i in range(len(hist) - 1))


def test_memory_size_is_reported_and_respected():
    res = lbfgs_minimize(lambda x: sum(v * v for v in x), [1.0] * 5,
                         lambda x: [2.0 * v for v in x], m=3)
    assert res["memory"] == 3


def test_validation():
    for call in (lambda: lbfgs_minimize(lambda x: 0.0, [], lambda x: []),
                 lambda: lbfgs_minimize(lambda x: x[0] ** 2, [1.0],
                                        lambda x: [2 * x[0]], m=0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
