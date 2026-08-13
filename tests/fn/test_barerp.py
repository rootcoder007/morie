"""Tests for barerp (Frisch 1956 log potential; Boyd ch. 11 barrier method)."""

import math

import pytest

from morie.fn.barerp import (_as_fun, barrier_lp, barrier_method,
                             central_path_dual, central_point,
                             centering_steps, frisch_potential, log_barrier,
                             log_barrier_gradient, log_barrier_hessian,
                             phase1)

# minimise c'x subject to 0 <= x <= 1 -- separable, so the central path
# is a closed-form root of  t c x^2 - (t c + 2) x + 1 = 0.
C = [1.0, -2.0, 0.35]
N = len(C)
M_CON = 2 * N


def _exact_central(ci, t):
    a = t * ci
    if abs(a) < 1e-14:
        return 0.5
    disc = math.sqrt(a * a + 4.0)
    for r in ((a + 2.0 - disc) / (2.0 * a), (a + 2.0 + disc) / (2.0 * a)):
        if 0.0 < r < 1.0:
            return r
    raise AssertionError("no root in (0, 1)")


def _box():
    obj = {"f": lambda z: sum(C[j] * z[j] for j in range(N)),
           "grad": lambda z: list(C), "affine": True}
    cons = []
    for j in range(N):
        cons.append({"f": (lambda z, _j=j: -z[_j]),
                     "grad": (lambda z, _j=j: [-1.0 if k == _j else 0.0
                                               for k in range(N)]),
                     "affine": True})
        cons.append({"f": (lambda z, _j=j: z[_j] - 1.0),
                     "grad": (lambda z, _j=j: [1.0 if k == _j else 0.0
                                               for k in range(N)]),
                     "affine": True})
    return obj, cons


OBJ, CONS = _box()


# --------------------------------------------------------------------------
# Frisch eq. 5.1 and Boyd eq. 11.5
# --------------------------------------------------------------------------

def test_the_frisch_potential_is_minus_the_log_barrier():
    fv = [-0.4, -1.25, -3.0]
    assert frisch_potential([-v for v in fv]) == pytest.approx(
        -log_barrier(fv), abs=1e-14)
    assert log_barrier(fv) == pytest.approx(
        -sum(math.log(-v) for v in fv), abs=1e-14)


def test_the_barrier_is_infinite_outside_the_strict_interior():
    assert log_barrier([-1.0, 0.0]) == float("inf")     # on the boundary
    assert log_barrier([-1.0, 0.5]) == float("inf")     # outside
    assert frisch_potential([1.0, 0.0]) == float("-inf")
    assert frisch_potential([1.0, -0.5]) == float("-inf")


def _f(z):
    return [z[0] ** 2 + z[1] - 3.0, -z[0] - 1.0, z[1] - 2.0]


def _jac(z):
    return [[2.0 * z[0], 1.0], [-1.0, 0.0], [0.0, 1.0]]


def _hs(z):
    return [[[2.0, 0.0], [0.0, 0.0]], None, None]


def test_barrier_gradient_matches_a_difference_of_the_barrier():
    x, h = [0.4, 0.9], 1e-6
    g = log_barrier_gradient(_f(x), _jac(x))
    for j in range(2):
        up, dn = list(x), list(x)
        up[j] += h
        dn[j] -= h
        num = (log_barrier(_f(up)) - log_barrier(_f(dn))) / (2 * h)
        assert g[j] == pytest.approx(num, abs=1e-6)


def test_barrier_hessian_matches_a_difference_of_the_gradient():
    x, h = [0.4, 0.9], 1e-6
    H = log_barrier_hessian(_f(x), _jac(x), _hs(x))
    for a in range(2):
        for b in range(2):
            up, dn = list(x), list(x)
            up[b] += h
            dn[b] -= h
            num = (log_barrier_gradient(_f(up), _jac(up))[a]
                   - log_barrier_gradient(_f(dn), _jac(dn))[a]) / (2 * h)
            assert H[a][b] == pytest.approx(num, abs=1e-5)
    assert H[0][1] == pytest.approx(H[1][0], abs=1e-14)


def test_the_second_barrier_hessian_term_is_dropped_only_when_affine():
    x = [0.4, 0.9]
    with_h = log_barrier_hessian(_f(x), _jac(x), _hs(x))
    without = log_barrier_hessian(_f(x), _jac(x), None)
    # constraint 1 is not affine, so the nabla^2 f_i term must matter
    assert abs(with_h[0][0] - without[0][0]) > 1e-6


# --------------------------------------------------------------------------
# the central path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("t", [0.5, 2.0, 17.0, 400.0])
def test_the_central_point_is_the_closed_form_root(t):
    fo = _as_fun(OBJ)
    co = [_as_fun(c) for c in CONS]
    # tol here is what we ask of Newton, not what counts as a match
    xs, _, _ = central_point(fo, co, [0.5] * N, t, tol=1e-18, max_iter=400)
    for j in range(N):
        assert xs[j] == pytest.approx(_exact_central(C[j], t), abs=1e-9)


def test_the_dual_point_is_positive_and_certifies_a_gap_of_m_over_t():
    res = barrier_method(OBJ, CONS, [0.5] * N, t0=1.0, mu=10.0, eps=1e-9)
    lam = res["lambda_"]
    assert all(v > 0.0 for v in lam)
    fvals = [-v for v in res["slack"]]
    assert sum(-lam[i] * fvals[i] for i in range(M_CON)) == pytest.approx(
        M_CON / res["t"], rel=1e-9)
    assert res["gap"] == pytest.approx(M_CON / res["t"], rel=1e-15)
    # and the gap really bounds the error against the exact optimum
    p_star = sum(min(ci, 0.0) for ci in C)
    assert res["fun"] - p_star <= res["gap"] + 1e-12
    assert res["gap"] < 1e-9
    for j in range(N):
        assert res["x"][j] == pytest.approx(1.0 if C[j] < 0 else 0.0,
                                            abs=1e-8)


def test_central_path_dual_is_the_printed_formula():
    lam = central_path_dual([-0.5, -2.0], 4.0)
    assert lam[0] == pytest.approx(1.0 / (4.0 * 0.5))
    assert lam[1] == pytest.approx(1.0 / (4.0 * 2.0))
    with pytest.raises(ValueError):
        central_path_dual([-1.0], 0.0)


# --------------------------------------------------------------------------
# Algorithm 11.1 and equation 11.13
# --------------------------------------------------------------------------

@pytest.mark.parametrize("t0,mu,eps", [(1.0, 10.0, 1e-6), (1.0, 2.0, 1e-6),
                                       (5.0, 50.0, 1e-9), (0.1, 20.0, 1e-3)])
def test_equation_11_13_predicts_the_outer_iteration_count(t0, mu, eps):
    r = barrier_method(OBJ, CONS, [0.5] * N, t0=t0, mu=mu, eps=eps)
    assert r["outer"] == centering_steps(M_CON, eps, t0, mu) + 1
    assert r["gap"] < eps
    assert r["converged"]
    gaps = [h[1] for h in r["history"]]
    for i in range(len(gaps) - 1):
        assert gaps[i] / gaps[i + 1] == pytest.approx(mu, rel=1e-12)


def test_centering_steps_validation():
    assert centering_steps(6, 1e-6, 1.0, 10.0) == 7
    with pytest.raises(ValueError):
        centering_steps(6, 1e-6, 1.0, 1.0)
    with pytest.raises(ValueError):
        centering_steps(6, 0.0, 1.0, 10.0)


def test_all_three_centering_routes_reach_the_same_point():
    base = barrier_method(OBJ, CONS, [0.5] * N, eps=1e-6)
    none = barrier_method(OBJ, CONS, [0.5] * N, eps=1e-6, centering="none")
    grad = barrier_method(OBJ, CONS, [0.5] * N, eps=1e-3, mu=10.0,
                          centering="gradient", max_inner=4000, tol=1e-12)
    for j in range(N):
        assert none["x"][j] == pytest.approx(base["x"][j], abs=1e-5)
        assert grad["x"][j] == pytest.approx(
            _exact_central(C[j], grad["t"]), abs=1e-4)
    # Newton is why the gradient route was displaced
    assert grad["newton"] > 20 * base["newton"]
    assert none["outer"] == 1


def test_barrier_method_validation():
    with pytest.raises(ValueError, match="centering"):
        barrier_method(OBJ, CONS, [0.5] * N, centering="magic")
    with pytest.raises(ValueError, match="mu"):
        barrier_method(OBJ, CONS, [0.5] * N, mu=1.0)
    with pytest.raises(ValueError, match="positive"):
        barrier_method(OBJ, CONS, [0.5] * N, eps=0.0)
    with pytest.raises(ValueError, match="strictly feasible"):
        barrier_method(OBJ, CONS, [2.0] * N)
    with pytest.raises(ValueError, match="no inequality"):
        barrier_method(OBJ, [], [0.5] * N)


# --------------------------------------------------------------------------
# phase I and equalities
# --------------------------------------------------------------------------

_TRI = [{"f": lambda z: -z[0], "grad": lambda z: [-1.0, 0.0],
         "affine": True},
        {"f": lambda z: -z[1], "grad": lambda z: [0.0, -1.0],
         "affine": True},
        {"f": lambda z: z[0] + z[1] - 1.0, "grad": lambda z: [1.0, 1.0],
         "affine": True}]


def test_phase1_finds_a_strictly_feasible_point():
    ph = phase1(_TRI, [5.0, 7.0])          # start far outside
    assert ph["feasible"]
    assert ph["s"] < 0.0
    assert all(_as_fun(c).val(ph["x"]) < 0.0 for c in _TRI)


def test_phase1_reports_an_empty_set_as_infeasible():
    empty = [{"f": lambda z: z[0] + 1.0, "grad": lambda z: [1.0],
              "affine": True},
             {"f": lambda z: 1.0 - z[0], "grad": lambda z: [-1.0],
              "affine": True}]
    ph = phase1(empty, [0.0])
    assert not ph["feasible"]
    assert ph["s"] >= 0.0


def test_barrier_lp_finds_its_own_starting_point():
    lp = barrier_lp([1.0, 1.0],
                    [[-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0]],
                    [0.0, 0.0, -1.0], eps=1e-8)
    assert lp["converged"]
    assert lp["fun"] == pytest.approx(1.0, abs=1e-6)
    assert sum(lp["x"]) == pytest.approx(1.0, abs=1e-6)


def test_barrier_lp_shape_validation():
    with pytest.raises(ValueError, match="b_ub"):
        barrier_lp([1.0], [[1.0], [1.0]], [1.0])
    with pytest.raises(ValueError, match="row width"):
        barrier_lp([1.0, 1.0], [[1.0]], [1.0])


def test_equality_constraints_are_held_exactly():
    n = 4
    qobj = {"f": lambda z: sum(v * v for v in z),
            "grad": lambda z: [2.0 * v for v in z],
            "hess": lambda z: [[2.0 if a == b else 0.0 for b in range(n)]
                               for a in range(n)]}
    qcons = [{"f": (lambda z, _j=j: -z[_j]),
              "grad": (lambda z, _j=j: [-1.0 if k == _j else 0.0
                                        for k in range(n)]),
              "affine": True} for j in range(n)]
    r = barrier_method(qobj, qcons, [0.4, 0.3, 0.2, 0.1],
                       aeq=[[1.0] * n], beq=[1.0], eps=1e-10, mu=15.0)
    assert sum(r["x"]) == pytest.approx(1.0, abs=1e-12)
    for v in r["x"]:
        assert v == pytest.approx(1.0 / n, abs=1e-7)
    assert r["fun"] == pytest.approx(1.0 / n, abs=1e-9)
    with pytest.raises(ValueError, match="equality row"):
        barrier_method(qobj, qcons, [0.9, 0.3, 0.2, 0.1],
                       aeq=[[1.0] * n], beq=[1.0])


def test_numerical_derivatives_are_used_when_none_are_supplied():
    """A bare callable must still work -- everything is differenced."""
    cons = [lambda z: -z[0], lambda z: z[0] - 1.0]
    r = barrier_method(lambda z: (z[0] - 0.25) ** 2, cons, [0.5],
                       eps=1e-8)
    assert r["x"][0] == pytest.approx(0.25, abs=1e-4)
