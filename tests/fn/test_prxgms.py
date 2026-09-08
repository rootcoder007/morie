"""Tests for prxgms (proximal gradient / FISTA).

Replaces the generated stub, which imported a name the module never had.
"""

from morie.fn.prxgms import lasso_fista, prox_gradient, soft_threshold


def test_soft_threshold_is_the_l1_prox():
    assert abs(soft_threshold([3.0], 1.0)[0] - 2.0) < 1e-12
    assert abs(soft_threshold([-3.0], 1.0)[0] + 2.0) < 1e-12
    assert abs(soft_threshold([0.5], 1.0)[0]) < 1e-12


def test_a_smooth_quadratic_is_minimised():
    def f(x):
        return (x[0] - 2.0) ** 2

    def g(x):
        return [2.0 * (x[0] - 2.0)]

    res = prox_gradient(f, g, lambda v, t: v, [0.0], L=2.0)
    assert abs(res["x"][0] - 2.0) < 1e-6
    assert res["converged"]


def test_lasso_sets_small_coefficients_to_zero():
    # y = 3 x1, with x2 and x3 irrelevant
    A = [[1.0, 0.5, -0.2], [2.0, -0.3, 0.1], [3.0, 0.2, 0.4],
         [4.0, -0.1, -0.3], [5.0, 0.4, 0.2]]
    b = [3.0 * row[0] for row in A]
    res = lasso_fista(A, b, lam=1.0)
    assert abs(res["x"][0] - 3.0) < 0.5
    assert abs(res["x"][1]) < 0.5 and abs(res["x"][2]) < 0.5


def test_a_larger_penalty_shrinks_more():
    A = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    b = [2.0, 3.0, 5.0]
    small = lasso_fista(A, b, lam=0.01)["x"]
    large = lasso_fista(A, b, lam=5.0)["x"]
    assert sum(abs(v) for v in large) < sum(abs(v) for v in small)


def test_a_huge_penalty_zeroes_everything():
    A = [[1.0, 0.0], [0.0, 1.0]]
    b = [1.0, 1.0]
    res = lasso_fista(A, b, lam=1000.0)
    assert all(abs(v) < 1e-9 for v in res["x"])


def test_acceleration_reaches_at_least_as_low_an_objective():
    A = [[1.0, 0.5], [0.5, 1.0], [0.2, 0.8]]
    b = [1.0, 2.0, 1.5]
    plain = lasso_fista(A, b, lam=0.1, accelerate=False, max_iter=60)
    fista = lasso_fista(A, b, lam=0.1, accelerate=True, max_iter=60)
    # "objective" is the history; "fun" is the value reached
    assert fista["fun"] <= plain["fun"] + 1e-9
    assert fista["accelerated"] is True
    assert len(fista["objective"]) >= 1


def test_validation():
    try:
        prox_gradient(lambda x: 0.0, lambda x: [0.0], lambda v, t: v,
                      [0.0], L=0.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
