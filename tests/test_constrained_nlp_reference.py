"""Goldfarb-Idnani QP (KKT certificates), simplex, SQP and SLP on Hock-Schittkowski problems."""

import math

import pytest

from morie.fn._qpcore import dot, goldfarb_idnani, simplex_standard
from morie.fn._rng import random_uniform
from morie.fn.slpmin import sequential_linear_programming
from morie.fn.sqpmin import sequential_quadratic_programming
from morie.fn.sqprg import sqp_optimize

U = [float(v) for v in random_uniform(20000, seed=9, stream=0)]


def hs071():
    f = lambda x: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]  # noqa: E731
    eq = [lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 - 40]
    ineq = (
        [lambda x: x[0] * x[1] * x[2] * x[3] - 25]
        + [lambda x, i=i: x[i] - 1 for i in range(4)]
        + [lambda x, i=i: 5 - x[i] for i in range(4)]
    )
    return f, eq, ineq


def feasible(C, b, meq):
    # phase-1 certificate: x = x+ - x-, inequality slacks, then simplex feasibility
    m, n = len(C), len(C[0])
    A = [list(C[i]) + [-v for v in C[i]] + [(-1.0 if j == i - meq else 0.0) for j in range(m - meq)] for i in range(m)]
    return simplex_standard([0.0] * (2 * n + m - meq), A, b)[1] != "infeasible"


def test_goldfarb_idnani_solutions_satisfy_kkt():
    k = 0
    solved = 0
    for t in range(40):
        n, m = 3 + t % 4, 2 + t % 5
        meq = t % 3 if t % 3 < m else 0
        L = [[(U[k + i * n + j] - 0.5) * 2 if j <= i else 0.0 for j in range(n)] for i in range(n)]
        k += n * n
        G = [[dot(L[i], L[j]) + (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
        a = [(U[k + i] - 0.5) * 4 for i in range(n)]
        k += n
        C = [[(U[k + i * n + j] - 0.5) * 2 for j in range(n)] for i in range(m)]
        k += m * n
        b = [(U[k + i] - 0.5) * 2 for i in range(m)]
        k += m
        try:
            x, u, act = goldfarb_idnani(G, a, C, b, meq)
        except ValueError:
            assert not feasible(C, b, meq)
            continue
        assert feasible(C, b, meq)
        solved += 1
        s = [dot(C[i], x) - b[i] for i in range(m)]
        assert all(abs(s[i]) < 1e-9 for i in range(meq)) and all(s[i] > -1e-9 for i in range(meq, m))
        assert all(u[i] >= -1e-12 for i in range(meq, m)) and all(abs(u[i] * s[i]) < 1e-9 for i in range(m))
        assert max(abs(dot(G[j], x) + a[j] - sum(u[i] * C[i][j] for i in range(m))) for j in range(n)) < 1e-9
    assert solved >= 30
    with pytest.raises(ValueError):
        goldfarb_idnani([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], [[1.0, 0.0], [1.0, 0.0]], [0.0, 1.0], meq=2)
    x, u, _ = goldfarb_idnani([[2.0, 0.0], [0.0, 2.0]], [-2.0, -5.0], [[1.0, 1.0], [1.0, 1.0]], [3.0, 3.0], meq=2)
    assert abs(x[0] - 0.75) < 1e-12 and abs(x[1] - 2.25) < 1e-12  # a repeated equality is consistent


def test_simplex_standard_form():
    # min -x - 2y s.t. x + y + s1 = 4, x + s2 = 3, y + s3 = 3: optimum (1, 3), value -7
    x, st = simplex_standard([-1, -2, 0, 0, 0], [[1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]], [4, 3, 3])
    assert st == "optimal" and [round(v, 12) for v in x[:2]] == [1.0, 3.0]
    assert simplex_standard([-1, 0], [[1, -1]], [0])[1] == "unbounded"
    assert simplex_standard([1, 0], [[1, 1], [1, 1]], [1, 2])[1] == "infeasible"
    x, st = simplex_standard(
        [-1, -2, 0, 0, 1, 0], [[1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0], [-2, -2, 0, 0, 1, -1]], [1, 1, -4]
    )
    assert st == "optimal" and x == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # degenerate: every optimal variable at a bound


def test_sqp_hock_schittkowski():
    f, eq, ineq = hs071()
    r = sequential_quadratic_programming(f, [1, 5, 5, 1], eq=eq, ineq=ineq)
    assert r["converged"] and r["kkt_residual"] <= 1e-8 and r["violation"] <= 1e-8
    assert abs(r["fun"] - 17.0140173) < 1e-6  # HS071 optimum (Hock and Schittkowski 1981)
    assert max(abs(a - b) for a, b in zip(r["x"], [1.0, 4.7429994, 3.8211503, 1.3794082])) < 1e-6
    r = sequential_quadratic_programming(
        lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2,
        [0.0, 0.0],
        ineq=[lambda x: 1 - x[0] ** 2 - x[1] ** 2],
    )
    x, lam = r["x"], r["multipliers_ineq"][0]
    g = [-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)]
    assert (
        abs(x[0] ** 2 + x[1] ** 2 - 1) < 1e-10
        and lam > 0
        and max(abs(g[0] + 2 * lam * x[0]), abs(g[1] + 2 * lam * x[1])) < 1e-7
    )
    r = sequential_quadratic_programming(lambda x: (1 - x[0]) ** 2, [-1.2, 1.0], eq=[lambda x: 10 * (x[1] - x[0] ** 2)])
    assert max(abs(v - 1) for v in r["x"]) < 1e-9  # HS006


def test_slp_vertex_and_curved_solutions():
    r = sequential_linear_programming(
        lambda x: (x[0] - 5) ** 2 + (x[1] - 5) ** 2, [0.0, 0.0], ineq=[lambda x: 1 - x[0], lambda x: 2 - x[1]]
    )
    assert r["converged"] and abs(r["x"][0] - 1) < 1e-9 and r["x"][1] == 2.0 and r["n_iter"] == 9
    r = sequential_linear_programming(
        lambda x: -x[0] - 2 * x[1], [0.0, 0.0], ineq=[lambda x: 4 - x[0] ** 2 - x[1] ** 2]
    )
    assert abs(r["fun"] + math.sqrt(20)) < 1e-8 and abs(r["x"][0] - 2 / math.sqrt(5)) < 1e-4 and r["n_iter"] == 34
    f, eq, ineq = hs071()
    r = sequential_linear_programming(f, [1, 5, 5, 1], eq=eq, ineq=ineq)
    assert r["converged"] and abs(r["fun"] - 17.0140173) < 1e-6 and r["violation"] < 1e-8


def test_sqp_optimize_is_the_new_sqp():
    r = sqp_optimize(
        lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2,
        lambda x: [2 * (x[0] - 1), 2 * (x[1] - 2.5)],
        [lambda x: x[0] + x[1] - 3],
        [0.0, 0.0],
    )
    # the constraint Jacobian is a forward difference (h = 1e-7): rounding error about eps / h
    assert abs(r.extra["x"][0] - 0.75) < 1e-8 and abs(r.extra["x"][1] - 2.25) < 1e-8 and r.extra["converged"]
    assert abs(r.value - 0.125) < 1e-8


def test_sqp_merit_line_search_shortens_the_path():
    # full SQP steps also converge here but take 68 and 61 iterations; the l1-merit backtracking takes 27 and 35
    rb = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2  # noqa: E731
    r = sequential_quadratic_programming(rb, [-1.2, 1.0], ineq=[lambda x: 1 - x[0] ** 2 - x[1] ** 2])
    assert r["converged"] and r["n_iter"] == 27
    r = sequential_quadratic_programming(rb, [-1.2, 1.0])
    assert r["converged"] and r["n_iter"] == 35 and max(abs(v - 1) for v in r["x"]) < 1e-8
