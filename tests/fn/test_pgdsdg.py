"""Projected gradient descent."""
import importlib

import pytest

P = importlib.import_module("morie.fn.pgdsdg")
np = importlib.import_module("morie.fn._array_core")

A = [0.5, 0.9, -0.2]
PROJ = [("simplex", P.project_simplex),
        ("nonneg", P.project_nonneg),
        ("ball", lambda z: P.project_ball(z, 1.0)),
        ("box", lambda z: P.project_box(z, [-1.0] * 3, [1.0] * 3))]


def d2(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a)))


def test_the_simplex_projection_is_nearest():
    p = P.project_simplex(A)
    assert sum(p) == pytest.approx(1.0)
    assert min(p) >= 0.0
    rng = np.random.default_rng(7)
    for _ in range(5000):
        q = [rng.random() for _ in range(3)]
        s = sum(q)
        q = [v / s for v in q]
        assert d2(q, A) >= d2(p, A) - 1e-12


def test_it_is_not_clip_and_renormalise():
    p = P.project_simplex(A)
    naive = [max(0.0, v) for v in A]
    naive = [v / sum(naive) for v in naive]
    assert d2(naive, A) > d2(p, A) + 1e-9


@pytest.mark.parametrize("name,pr", PROJ)
def test_projections_are_non_expansive(name, pr):
    rng = np.random.default_rng(11)
    for _ in range(1500):
        u = [rng.random() * 6 - 3 for _ in range(3)]
        v = [rng.random() * 6 - 3 for _ in range(3)]
        assert d2(pr(u), pr(v)) <= d2(u, v) + 1e-9


@pytest.mark.parametrize("name,pr", PROJ)
def test_projections_are_idempotent(name, pr):
    for u in ([2.0, -1.0, 0.5], [0.1, 0.1, 0.1], [-4.0, 3.0, 2.0]):
        assert d2(pr(pr(u)), pr(u)) < 1e-20


@pytest.mark.parametrize("name,pr", PROJ)
def test_descent_reaches_the_projection_itself(name, pr):
    f = lambda z: d2(z, A)
    g = lambda z: [2.0 * (z[i] - A[i]) for i in range(3)]
    r = P.projected_gradient(f, g, [1.0, 0.0, 0.0], pr)
    want = pr(A)
    assert r["x"] == pytest.approx(want, abs=1e-8)
    assert r["fixed_point_residual"] < 1e-8
    assert r["converged"]


def test_an_optimum_outside_the_set_lands_on_the_boundary():
    r = P.projected_gradient(
        lambda z: d2(z, [5.0, 5.0, 5.0]),
        lambda z: [2.0 * (z[i] - 5.0) for i in range(3)],
        [0.0, 0.0, 0.0],
        lambda z: P.project_box(z, [-1.0] * 3, [1.0] * 3))
    assert r["x"] == pytest.approx([1.0, 1.0, 1.0])


Q = [1000.0, 1.0, 0.01]
FA = lambda z: sum(Q[i] * (z[i] - 0.3) ** 2 for i in range(3))
GA = lambda z: [2.0 * Q[i] * (z[i] - 0.3) for i in range(3)]


def test_fista_is_not_monotone_but_gets_further():
    ball = lambda z: P.project_ball(z, 2.0)
    nm = P.projected_gradient(FA, GA, [3.0] * 3, ball, rule="fista",
                              max_iter=300, tol=0.0)
    mo = P.projected_gradient(FA, GA, [3.0] * 3, ball,
                              rule="backtracking", max_iter=300,
                              tol=0.0)
    assert not nm["monotone"]
    assert mo["monotone"]
    assert nm["history"][-1] < mo["history"][-1]


def test_the_monotone_flag_matches_the_history():
    r = P.projected_gradient(FA, GA, [3.0] * 3, P.project_nonneg,
                             rule="fista", max_iter=100, tol=0.0)
    h = r["history"]
    assert r["monotone"] == all(h[i] >= h[i + 1] - 1e-12
                                for i in range(len(h) - 1))


def test_backtracking_finds_a_step_without_knowing_l():
    r = P.projected_gradient(
        lambda z: sum(100.0 * v * v for v in z),
        lambda z: [200.0 * v for v in z],
        [1.0, 1.0, 1.0], P.project_nonneg, rule="backtracking",
        max_iter=200, tol=0.0)
    assert r["n_backtracks"] > 0
    assert r["history"][-1] < r["history"][0]


def test_the_entry_point_matches():
    f = lambda z: d2(z, A)
    g = lambda z: [2.0 * (z[i] - A[i]) for i in range(3)]
    assert P.projected_gradient_descent(f, g, [1.0, 0.0, 0.0],
                                        P.project_simplex)["x"] \
        == pytest.approx(P.project_simplex(A), abs=1e-8)


@pytest.mark.parametrize("call", [
    lambda: P.projected_gradient(lambda z: 0.0, lambda z: [0.0] * 3,
                                 [1.0] * 3, P.project_nonneg,
                                 rule="adam"),
    lambda: P.projected_gradient(lambda z: 0.0, lambda z: [0.0] * 3,
                                 [1.0] * 3, P.project_nonneg,
                                 rule="fixed"),
    lambda: P.projected_gradient(lambda z: 0.0, lambda z: [0.0],
                                 [1.0] * 3, P.project_nonneg),
    lambda: P.project_ball([1.0], radius=0.0),
    lambda: P.project_ball([1.0, 2.0], centre=[0.0]),
    lambda: P.project_simplex([]),
    lambda: P.project_simplex([1.0], total=0.0),
    lambda: P.project_box([1.0, 2.0], [1.0], [2.0]),
    lambda: P.project_box([1.0], [2.0], [1.0]),
])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call()
