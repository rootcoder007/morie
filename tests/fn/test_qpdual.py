"""Tests for qpdual (Frank-Wolfe QP over a simplex or box).

Replaces the generated stub, which imported a name the module never had.
"""

from morie.fn.qpdual import frank_wolfe_qp


def test_the_unconstrained_optimum_inside_a_box_is_found():
    # min 1/2 x'Qx + c'x with Q = I, c = (-2, -3): optimum (2, 3)
    Q = [[1.0, 0.0], [0.0, 1.0]]
    c = [-2.0, -3.0]
    res = frank_wolfe_qp(Q, c, domain="box", lower=[-10.0, -10.0],
                         upper=[10.0, 10.0])
    assert abs(res["x"][0] - 2.0) < 1e-3
    assert abs(res["x"][1] - 3.0) < 1e-3


def test_a_box_bound_that_binds_is_respected():
    Q = [[1.0, 0.0], [0.0, 1.0]]
    c = [-20.0, -3.0]
    res = frank_wolfe_qp(Q, c, domain="box", lower=[0.0, 0.0],
                         upper=[5.0, 5.0])
    assert abs(res["x"][0] - 5.0) < 1e-6
    assert all(0.0 - 1e-9 <= v <= 5.0 + 1e-9 for v in res["x"])


def test_the_simplex_solution_is_a_distribution():
    Q = [[2.0, 0.5], [0.5, 1.0]]
    c = [-1.0, -2.0]
    res = frank_wolfe_qp(Q, c, domain="simplex")
    assert abs(sum(res["x"]) - 1.0) < 1e-6
    assert all(v >= -1e-9 for v in res["x"])


def test_the_objective_never_increases():
    Q = [[2.0, 0.5], [0.5, 1.0]]
    c = [-1.0, -2.0]
    res = frank_wolfe_qp(Q, c, domain="simplex")
    hist = res["history"]
    assert all(hist[i + 1] <= hist[i] + 1e-9
               for i in range(len(hist) - 1))


def test_the_duality_gap_closes():
    Q = [[2.0, 0.0], [0.0, 2.0]]
    c = [-1.0, -1.0]
    res = frank_wolfe_qp(Q, c, domain="simplex", tol=1e-10)
    assert res["gap"] < 1e-6
    assert res["converged"]


def test_the_reported_value_matches_the_point():
    Q = [[2.0, 0.5], [0.5, 1.0]]
    c = [-1.0, -2.0]
    res = frank_wolfe_qp(Q, c, domain="simplex")
    x = res["x"]
    want = 0.5 * sum(x[i] * Q[i][j] * x[j] for i in range(2)
                     for j in range(2)) + sum(c[i] * x[i]
                                              for i in range(2))
    assert abs(res["fun"] - want) < 1e-9


def test_validation():
    Q = [[1.0, 0.0], [0.0, 1.0]]
    for call in (lambda: frank_wolfe_qp([[1.0, 0.0]], [0.0, 0.0]),
                 lambda: frank_wolfe_qp(Q, [0.0]),
                 lambda: frank_wolfe_qp(Q, [0.0, 0.0], domain="ball"),
                 lambda: frank_wolfe_qp(Q, [0.0, 0.0], step="newton")):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
