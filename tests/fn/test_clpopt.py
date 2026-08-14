"""Two-phase primal simplex."""
import importlib

import pytest

C = importlib.import_module("morie.fn.clpopt")

WA = [[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]]
WB = [4.0, 12.0, 18.0]
WC = [3.0, 5.0]

BC = [-0.75, 150.0, -0.02, 6.0, 0.0, 0.0, 0.0]
BA = [[0.25, -60.0, -0.04, 9.0, 1.0, 0.0, 0.0],
      [0.5, -90.0, -0.02, 3.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
BB = [0.0, 0.0, 1.0]


def test_the_textbook_optimum():
    r = C.linprog(WC, WA, WB, maximise=True)
    assert r["status"] == "optimal"
    assert r["x"] == pytest.approx([2.0, 6.0])
    assert r["fun"] == pytest.approx(36.0)


def test_the_shadow_prices():
    r = C.linprog(WC, WA, WB, maximise=True)
    assert r["duals"] == pytest.approx([0.0, 1.5, 1.0])


def test_strong_duality_is_exact():
    r = C.linprog(WC, WA, WB, maximise=True)
    assert sum(WB[i] * r["duals"][i] for i in range(3)) \
        == pytest.approx(r["fun"], abs=1e-12)


def test_complementary_slackness():
    r = C.linprog(WC, WA, WB, maximise=True)
    for i in range(3):
        slack = WB[i] - sum(WA[i][j] * r["x"][j] for j in range(2))
        assert slack * r["duals"][i] == pytest.approx(0.0, abs=1e-12)
        assert r["slack"][i] == pytest.approx(slack, abs=1e-12)


def test_dantzig_cycles_on_beale():
    r = C.simplex(BC, BA, BB, "dantzig", initial_basis=[4, 5, 6])
    assert r["status"] == "cycling"
    assert "bland" in r["message"]


def test_bland_does_not():
    r = C.simplex(BC, BA, BB, "bland", initial_basis=[4, 5, 6])
    assert r["status"] == "optimal"
    assert r["fun"] == pytest.approx(-0.05)


def test_an_unbounded_program():
    assert C.linprog([-1.0, -1.0], [[1.0, -1.0]], [1.0])["status"] \
        == "unbounded"


def test_an_infeasible_program():
    r = C.linprog([1.0, 1.0], [[1.0, 1.0], [-1.0, -1.0]],
                  [1.0, -3.0])
    assert r["status"] == "infeasible"
    assert r["x"] is None


def test_a_transportation_problem_is_integral():
    sup, dem = [20.0, 30.0], [10.0, 25.0, 15.0]
    cost = [4.0, 6.0, 9.0, 5.0, 3.0, 8.0]
    aeq, beq = [], []
    for i in range(2):
        row = [0.0] * 6
        for j in range(3):
            row[3 * i + j] = 1.0
        aeq.append(row)
        beq.append(sup[i])
    for j in range(3):
        row = [0.0] * 6
        for i in range(2):
            row[3 * i + j] = 1.0
        aeq.append(row)
        beq.append(dem[j])
    r = C.linprog(cost, None, None, aeq, beq)
    assert r["status"] == "optimal"
    assert all(v == pytest.approx(round(v), abs=1e-9)
               for v in r["x"])
    for k in range(5):
        assert sum(aeq[k][j] * r["x"][j] for j in range(6)) \
            == pytest.approx(beq[k], abs=1e-9)


def test_multiple_optima_are_flagged():
    r = C.linprog([1.0, 1.0], [[-1.0, -1.0]], [-2.0])
    assert r["status"] == "optimal"
    assert r["multiple_optima"]


def test_upper_bounds_are_honoured():
    r = C.linprog([-1.0, -1.0], [[1.0, 1.0]], [10.0],
                  upper=[3.0, 4.0])
    assert r["x"] == pytest.approx([3.0, 4.0])
    assert r["fun"] == pytest.approx(-7.0)


def test_standard_form_makes_the_rhs_non_negative():
    sf = C.standard_form([1.0, 1.0], [[-1.0, -1.0]], [-2.0])
    assert all(v >= 0 for v in sf["b"])
    assert sf["n_original"] == 2 and sf["n_slack"] == 1


@pytest.mark.parametrize("call", [
    lambda: C.linprog([], [[1.0]], [1.0]),
    lambda: C.linprog([1.0, 1.0]),
    lambda: C.linprog([1.0, 1.0], [[1.0, 1.0]], [1.0, 2.0]),
    lambda: C.linprog([1.0, 1.0], [[1.0, 1.0], [1.0]], [1.0, 2.0]),
    lambda: C.simplex([1.0], [[1.0]], [-1.0]),
    lambda: C.simplex([1.0], [[1.0]], [1.0], "steepest"),
    lambda: C.simplex(BC, BA, BB, "bland", initial_basis=[4, 5]),
    lambda: C.simplex(BC, BA, BB, "bland", initial_basis=[4, 4, 6]),
    lambda: C.simplex(BC, BA, BB, "bland", initial_basis=[0, 1, 2]),
])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call()
