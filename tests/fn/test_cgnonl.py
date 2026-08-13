"""Tests for cgnonl (Fletcher & Reeves 1964, nonlinear conjugate gradients)."""

import importlib
import math

import pytest

from morie.fn import _array_core as np

M = importlib.import_module("morie.fn.cgnonl")


def _spd(n, seed=3):
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            v = math.sin(1.0 + (i + 1) * (j + 2) * seed) * 0.5
            A[i][j] += v
            A[j][i] += v
    for i in range(n):
        A[i][i] += n + 2.0 + i
    return A


def _quad(A, b):
    n = len(b)

    def f(x):
        ax = [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        return 0.5 * sum(x[i] * ax[i] for i in range(n)) \
            - sum(b[i] * x[i] for i in range(n))

    def g(x):
        return [sum(A[i][j] * x[j] for j in range(n)) - b[i]
                for i in range(n)]

    def hv(p):
        return [sum(A[i][j] * p[j] for j in range(n)) for i in range(n)]

    return f, g, hv


def rosen(x):
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def rosen_g(x):
    return [-400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2)]


# --------------------------------------------------------------------------
# the paper's guarantee
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_a_quadratic_is_solved_exactly_within_n_iterations(n):
    """"guaranteed ... to locate the minimum of any quadratic function
    of n arguments in at most n iterations"."""
    A = _spd(n)
    b = [1.0 + 0.3 * i for i in range(n)]
    f, g, hv = _quad(A, b)
    xs = np.linalg.solve(np.asarray(A, dtype=float),
                         np.asarray(b, dtype=float))
    r = M.nonlinear_cg(f, g, [0.0] * n, line_search="exact-quadratic",
                       hess_vec=hv, restart=0, tol=1e-14)
    assert r["n_iter"] <= n
    assert r["converged"]
    for i in range(n):
        assert r["x"][i] == pytest.approx(float(xs[i]), abs=1e-9)


def test_conjugacy_and_gradient_orthogonality():
    n = 6
    A = _spd(n, seed=5)
    b = [0.7 * (i + 1) for i in range(n)]
    f, g, hv = _quad(A, b)
    x = [0.0] * n
    gi = g(x)
    p = [-v for v in gi]
    ps, gs = [list(p)], [list(gi)]
    for _ in range(n - 1):
        ap = hv(p)
        t = -sum(p[i] * gi[i] for i in range(n)) \
            / sum(p[i] * ap[i] for i in range(n))
        x = [x[i] + t * p[i] for i in range(n)]
        g_new = g(x)
        beta = M.beta_fletcher_reeves(g_new, gi)
        p = [-g_new[i] + beta * p[i] for i in range(n)]
        gi = g_new
        ps.append(list(p))
        gs.append(list(gi))
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            api = hv(ps[j])
            assert abs(sum(ps[i][k] * api[k] for k in range(n))) < 1e-8
            assert abs(sum(gs[i][k] * gs[j][k] for k in range(n))) < 1e-8


# --------------------------------------------------------------------------
# the beta rules
# --------------------------------------------------------------------------

def test_beta_formulas_are_the_printed_ones():
    ga = [0.3, -1.2, 0.5]
    gb = [-0.4, 0.9, 2.0]
    assert M.beta_fletcher_reeves(gb, ga) == pytest.approx(
        sum(v * v for v in gb) / sum(v * v for v in ga), rel=1e-14)
    assert M.beta_polak_ribiere(gb, ga) == pytest.approx(
        sum(gb[i] * (gb[i] - ga[i]) for i in range(3))
        / sum(v * v for v in ga), rel=1e-14)
    assert M.beta_fletcher_reeves(gb, [0.0, 0.0, 0.0]) == 0.0
    assert M.beta_polak_ribiere(gb, [0.0, 0.0, 0.0]) == 0.0


def test_the_polak_ribiere_plus_safeguard():
    small, big = [0.01, 0.0, 0.0], [1.0, 0.0, 0.0]
    assert M.beta_polak_ribiere(small, big) < 0.0
    assert M.beta_polak_ribiere(small, big, plus=True) == 0.0


def test_fletcher_reeves_and_polak_ribiere_agree_on_a_quadratic():
    """Successive gradients are orthogonal there, so the two coincide.

    That cross-checks both formulas at once.
    """
    n = 6
    A = _spd(n, seed=5)
    b = [0.7 * (i + 1) for i in range(n)]
    f, g, hv = _quad(A, b)
    a = M.nonlinear_cg(f, g, [0.0] * n, beta="fletcher-reeves",
                       line_search="exact-quadratic", hess_vec=hv,
                       restart=0, tol=1e-14)
    c = M.nonlinear_cg(f, g, [0.0] * n, beta="polak-ribiere",
                       line_search="exact-quadratic", hess_vec=hv,
                       restart=0, tol=1e-14)
    for i in range(n):
        assert a["x"][i] == pytest.approx(c["x"][i], abs=1e-10)


# --------------------------------------------------------------------------
# the line search
# --------------------------------------------------------------------------

def test_the_exact_quadratic_step_is_the_line_minimum():
    n = 4
    A = _spd(n, seed=2)
    b = [1.0] * n
    f, g, hv = _quad(A, b)
    x = [0.2] * n
    gx = g(x)
    p = [-v for v in gx]
    t = M._exact_quadratic_step(x, p, gx, hv)
    ap = hv(p)
    assert t == pytest.approx(
        -sum(p[i] * gx[i] for i in range(n))
        / sum(p[i] * ap[i] for i in range(n)), rel=1e-14)
    base = f([x[i] + t * p[i] for i in range(n)])
    for d in (-1e-4, 1e-4):
        assert f([x[i] + (t + d) * p[i] for i in range(n)]) > base


def test_cubic_interpolation_finds_a_known_minimum():
    def q(t):
        return t ** 3 - 3.0 * t ** 2 + 1.0        # minimum at t = 2

    def dq(t):
        return 3.0 * t ** 2 - 6.0 * t

    got = M.cubic_interpolate(0.5, q(0.5), dq(0.5), 3.0, q(3.0), dq(3.0))
    assert got == pytest.approx(2.0, abs=1e-9)
    # a degenerate bracket must not produce a point outside it
    assert M.cubic_interpolate(1.0, 0.0, 0.0, 1.0, 0.0, 0.0) == 1.0


def test_the_line_search_refuses_a_non_descent_direction():
    # at (0, 0) the gradient is (-2, 0), so (-1, 0) is the ascent one
    with pytest.raises(ValueError, match="descent"):
        M.line_search_fr(rosen, rosen_g, [0.0, 0.0], [-1.0, 0.0],
                         rosen([0.0, 0.0]), rosen_g([0.0, 0.0]))
    with pytest.raises(ValueError):
        M.line_search_fr(rosen, rosen_g, [0.0, 0.0], [0.0, 0.0],
                         rosen([0.0, 0.0]), rosen_g([0.0, 0.0]))


def test_the_line_search_decreases_f():
    x = [-1.2, 1.0]
    g0 = rosen_g(x)
    p = [-v for v in g0]
    t, xn, fn, gn, ev = M.line_search_fr(rosen, rosen_g, x, p, rosen(x),
                                         g0, est=0.0)
    assert t > 0.0
    assert fn < rosen(x)
    assert ev >= 1


# --------------------------------------------------------------------------
# Rosenbrock and restarts
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rule", list(M.BETA_RULES))
def test_rosenbrocks_banana_valley(rule):
    """The paper's own Table 1 function, the one that motivated restarts."""
    r = M.nonlinear_cg(rosen, rosen_g, [-1.2, 1.0], beta=rule, est=0.0,
                       tol=1e-12, max_iter=8000)
    assert r["converged"]
    assert r["x"][0] == pytest.approx(1.0, abs=1e-4)
    assert r["x"][1] == pytest.approx(1.0, abs=1e-4)
    assert r["fun"] < 1e-12


def test_the_restart_cadence():
    r = M.nonlinear_cg(rosen, rosen_g, [-1.2, 1.0], est=0.0, restart=3,
                       tol=1e-12, max_iter=8000)
    assert r["restart_every"] == 3
    assert r["n_restart"] >= r["n_iter"] // 3
    # the default is n + 1, which is the paper's choice
    d = M.nonlinear_cg(rosen, rosen_g, [-1.2, 1.0], est=0.0, tol=1e-10)
    assert d["restart_every"] == 3
    # restart = 0 disables them
    n = 5
    A = _spd(n)
    b = [1.0] * n
    f, g, hv = _quad(A, b)
    z = M.nonlinear_cg(f, g, [0.0] * n, line_search="exact-quadratic",
                       hess_vec=hv, restart=0, tol=1e-14)
    assert z["n_restart"] == 0


def test_restarts_no_more_frequent_than_n_keep_quadratic_termination():
    n = 8
    A = _spd(n, seed=7)
    b = [1.0] * n
    f, g, hv = _quad(A, b)
    r = M.nonlinear_cg(f, g, [0.0] * n, line_search="exact-quadratic",
                       hess_vec=hv, restart=n, tol=1e-14)
    assert r["n_iter"] <= n
    assert r["converged"]


def test_an_already_optimal_start_does_no_work():
    n = 3
    A = _spd(n)
    b = [1.0] * n
    f, g, hv = _quad(A, b)
    xs = np.linalg.solve(np.asarray(A, dtype=float),
                         np.asarray(b, dtype=float))
    r = M.nonlinear_cg(f, g, [float(xs[i]) for i in range(n)],
                       line_search="exact-quadratic", hess_vec=hv)
    assert r["n_iter"] == 0
    assert r["converged"]


def test_validation():
    with pytest.raises(ValueError, match="beta"):
        M.nonlinear_cg(rosen, rosen_g, [0.0, 0.0], beta="magic")
    with pytest.raises(ValueError, match="line_search"):
        M.nonlinear_cg(rosen, rosen_g, [0.0, 0.0], line_search="magic")
    with pytest.raises(ValueError, match="hess_vec"):
        M.nonlinear_cg(rosen, rosen_g, [0.0, 0.0],
                       line_search="exact-quadratic")
    with pytest.raises(ValueError, match="empty"):
        M.nonlinear_cg(rosen, rosen_g, [])
    with pytest.raises(ValueError, match="restart"):
        M.nonlinear_cg(rosen, rosen_g, [0.0, 0.0], restart=-1)
    with pytest.raises(ValueError, match="max_iter"):
        M.nonlinear_cg(rosen, rosen_g, [0.0, 0.0], max_iter=0)
