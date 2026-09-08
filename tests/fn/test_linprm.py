"""Linear programming: simplex against interior point."""
import importlib

import pytest

L = importlib.import_module("morie.fn.linprm")

WA = [[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]]
WB = [4.0, 12.0, 18.0]
WC = [3.0, 5.0]

PROBLEMS = [
    (WC, WA, WB, True),
    ([1.0, 1.0], [[-1.0, -2.0], [-3.0, -1.0]], [-4.0, -6.0], False),
    ([2.0, 3.0, 4.0], [[1.0, 1.0, 1.0], [-1.0, 0.0, -2.0]],
     [10.0, -4.0], False),
]


@pytest.mark.parametrize("c,a,b,mx", PROBLEMS)
def test_the_two_solvers_agree(c, a, b, mx):
    s = L.solve_lp(c, a, b, method="simplex", maximise=mx)
    i = L.solve_lp(c, a, b, method="interior_point", maximise=mx)
    assert s["status"] == i["status"] == "optimal"
    assert s["fun"] == pytest.approx(i["fun"], abs=1e-7)


def test_the_interior_point_gap_closes():
    r = L.solve_lp(WC, WA, WB, method="interior_point",
                   maximise=True)
    assert r["gap"] < 1e-9
    assert r["primal_residual"] < 1e-9
    assert r["x"] == pytest.approx([2.0, 6.0], abs=1e-6)


def test_it_recovers_the_same_duals():
    r = L.solve_lp(WC, WA, WB, method="interior_point",
                   maximise=True)
    assert r["duals"] == pytest.approx([0.0, 1.5, 1.0], abs=1e-6)


def test_simplex_gives_a_vertex_of_a_flat_face():
    r = L.solve_lp([-1.0, -1.0], [[1.0, 1.0]], [1.0],
                   method="simplex")
    assert r["fun"] == pytest.approx(-1.0)
    assert min(abs(r["x"][0]), abs(r["x"][1])) < 1e-9


def test_interior_point_gives_its_analytic_centre():
    r = L.solve_lp([-1.0, -1.0], [[1.0, 1.0]], [1.0],
                   method="interior_point")
    assert r["fun"] == pytest.approx(-1.0, abs=1e-7)
    assert r["x"] == pytest.approx([0.5, 0.5], abs=1e-5)


def test_auto_is_simplex():
    assert L.solve_lp(WC, WA, WB, maximise=True)["solver"] \
        == "simplex"


def test_the_entry_point_dispatches():
    assert L.linear_program(WC, WA, WB, maximise=True)["fun"] \
        == pytest.approx(36.0)


def test_interior_point_on_a_bare_standard_form():
    r = L.interior_point([1.0, 1.0, 0.0], [[1.0, 1.0, 1.0]], [2.0])
    assert r["converged"]
    assert r["gap"] < 1e-9


@pytest.mark.parametrize("call", [
    lambda: L.solve_lp(WC, WA, WB, method="ellipsoid"),
    lambda: L.interior_point([1.0], [[1.0, 1.0]], [1.0]),
])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call()
