import math

import pytest

from morie.fn.compalgo import _order_distribution, shor_factoring, smt_solver


def _check_model(f, r):
    for k, (x, y, c) in enumerate(f["atoms"], start=1):
        if k in r.model:
            holds = r.solution[x] - r.solution[y] <= c
            assert holds == r.model[k]
    for cl in f["clauses"]:
        assert any(r.model.get(abs(v), True) == (v > 0) for v in cl)


def test_smt_sat_model_satisfies_constraints():
    f = {"atoms": [("x", "y", -1), ("y", "z", -1), ("z", "x", -1), ("z", "x", 5)], "clauses": [[1], [2], [3, 4]]}
    r = smt_solver(f)
    assert r.satisfiable and r.theory_conflicts == 1
    _check_model(f, r)


def test_smt_scheduling_disjunction():
    # two tasks of length 3 and 4 on one machine, both within [0, 7]
    f = {
        "atoms": [("s", "a", 0), ("s", "b", 0), ("a", "b", -3), ("b", "a", -4), ("a", "s", 4), ("b", "s", 3)],
        "clauses": [[1], [2], [3, 4], [5], [6]],
    }
    r = smt_solver(f)
    assert r.satisfiable
    _check_model(f, r)


def test_smt_unsat():
    r = smt_solver({"atoms": [("x", "y", 0), ("y", "x", -1)], "clauses": [[1], [2]]})
    assert not r.satisfiable
    assert r.learned == [[-1, -2]]


def test_order_distribution_sums_to_one_and_peaks():
    p = _order_distribution(4, 256)
    assert math.fsum(p) == pytest.approx(1.0, abs=1e-12)
    for y in (0, 64, 128, 192):
        assert p[y] == pytest.approx(0.25, abs=1e-12)
    p = _order_distribution(6, 2048)
    assert math.fsum(p) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("N", [15, 21, 35, 91, 143, 9, 27, 221])
def test_shor_factors(N):
    r = shor_factoring(N, seed=2)
    a, b = r.factors
    assert a * b == N and 1 < a < N
    if r.order is not None:
        assert pow(r.base, r.order, N) == 1


def test_shor_rejects():
    with pytest.raises(ValueError):
        shor_factoring(13)
    with pytest.raises(ValueError):
        shor_factoring(1147)
