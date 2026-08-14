"""Buchberger's algorithm and Groebner bases."""
import importlib
from fractions import Fraction as Fr

import pytest

G = importlib.import_module("morie.fn.groebn")
P = G.poly

F1 = P({(3, 0): 1, (1, 1): -2})
F2 = P({(2, 1): 1, (0, 2): -2, (1, 0): 1})
FSET = [F1, F2]
COMB = G.add(G.mul(P({(1, 0): 3, (0, 1): -1}), F1),
             G.mul(P({(0, 2): 2}), F2))


@pytest.mark.parametrize("order", G.ORDERS)
def test_all_s_polynomials_reduce_to_zero(order):
    B = G.buchberger(FSET, order)["basis"]
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            assert not G.normal_form(G.spoly(B[i], B[j], order), B,
                                     order)


def test_the_grlex_basis_is_the_published_one():
    # Cox, Little & O'Shea's worked example.
    assert G.buchberger(FSET, "grlex")["basis"] == [
        P({(2, 0): 1}), P({(1, 1): 1}),
        P({(0, 2): 1, (1, 0): Fr(-1, 2)})]


@pytest.mark.parametrize("p", [F1, F2, COMB])
def test_the_ideal_is_preserved(p):
    B = G.buchberger(FSET, "grlex")["basis"]
    assert not G.normal_form(p, B, "grlex")


def test_a_polynomial_outside_the_ideal_leaves_a_remainder():
    B = G.buchberger(FSET, "grlex")["basis"]
    assert G.normal_form(P({(0, 0): 1}), B, "grlex") != {}


def test_the_reduced_basis_is_unique_to_the_ideal():
    alt = [G.add(F1, F2), F2, G.mul(P({(1, 0): 1}), F1)]
    assert G.buchberger(alt, "grlex")["basis"] \
        == G.buchberger(FSET, "grlex")["basis"]


def test_the_order_changes_the_basis():
    assert G.buchberger(FSET, "lex")["basis"] \
        != G.buchberger(FSET, "grlex")["basis"]


def test_but_not_membership():
    probe = [COMB, F1, P({(0, 0): 1}), P({(1, 0): 1})]
    ans = [[G.ideal_member(p, FSET, o)["member"] for p in probe]
           for o in G.ORDERS]
    assert ans[0] == ans[1] == ans[2] == [True, True, False, False]


def test_division_reconstructs_its_dividend():
    B = G.buchberger(FSET, "grlex")["basis"]
    d = G.divide(COMB, B, "grlex")
    recon = d["remainder"]
    for q, g in zip(d["quotients"], B):
        recon = G.add(recon, G.mul(q, g))
    assert recon == COMB


def test_a_linear_system_reduces_to_its_solution():
    lin = [P({(1, 0, 0): 1, (0, 1, 0): 2, (0, 0, 1): 3,
              (0, 0, 0): -14}),
           P({(1, 0, 0): 2, (0, 1, 0): -1, (0, 0, 1): 1,
              (0, 0, 0): -3}),
           P({(1, 0, 0): 3, (0, 1, 0): 1, (0, 0, 1): -1,
              (0, 0, 0): -2})]
    B = G.buchberger(lin, "lex")["basis"]
    assert len(B) == 3
    sol = {}
    for p in B:
        sol[G.leading_monomial(p, "lex").index(1)] = \
            -p.get((0, 0, 0), Fr(0))
    assert (sol[0], sol[1], sol[2]) == (1, 2, 3)


def test_lex_eliminates_to_a_univariate_polynomial():
    circ = [P({(2, 0): 1, (0, 2): 1, (0, 0): -1}),
            P({(1, 0): 1, (0, 1): -1})]
    B = G.buchberger(circ, "lex")["basis"]
    uni = [p for p in B if all(e[0] == 0 for e in p)]
    assert uni == [P({(0, 2): 1, (0, 0): Fr(-1, 2)})]


def test_standard_monomials_count_the_solutions():
    sq = [P({(2, 0): 1, (0, 0): -1}), P({(0, 2): 1, (0, 0): -1})]
    B = G.buchberger(sq, "grlex")["basis"]
    lms = [G.leading_monomial(p, "grlex") for p in B]
    std = [(i, j) for i in range(6) for j in range(6)
           if not any(all(a <= b for a, b in zip(lm, (i, j)))
                      for lm in lms)]
    assert len(std) == 4


def test_an_inconsistent_system_collapses_to_one():
    assert G.buchberger([P({(1,): 1}), P({(1,): 1, (0,): -1})],
                        "lex")["basis"] == [P({(0,): 1})]


def test_the_coprimality_criterion_saves_work_only():
    a = G.buchberger(FSET, "grlex", prune=True)
    b = G.buchberger(FSET, "grlex", prune=False)
    assert a["basis"] == b["basis"]
    assert a["n_skipped"] > 0
    assert a["n_reductions"] < b["n_reductions"]


def test_the_reduced_basis_is_monic_and_irredundant():
    B = G.buchberger(FSET, "grlex")["basis"]
    assert all(G.leading_coeff(p, "grlex") == 1 for p in B)
    for i in range(len(B)):
        for j in range(len(B)):
            if i == j:
                continue
            lj = G.leading_monomial(B[j], "grlex")
            assert not any(all(u <= v for u, v in zip(lj, e))
                           for e in B[i])


def test_leading_pieces_of_the_zero_polynomial():
    assert G.leading_monomial({}, "lex") is None
    assert G.leading_coeff({}, "lex") == 0
    assert G.leading_term({}, "lex") == {}


def test_poly_drops_cancelling_terms():
    assert G.poly([((1, 0), 2), ((1, 0), -2)]) == {}


@pytest.mark.parametrize("call", [
    lambda: G.buchberger([], "lex"),
    lambda: G.buchberger(FSET, "deglex"),
    lambda: G.poly({(1, -1): 1}),
    lambda: G.poly({(1, 0): 1, (1,): 1}),
    lambda: G.divide(F1, [], "lex"),
    lambda: G.spoly({}, F1, "lex"),
])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call()


def test_the_entry_point_matches_buchberger():
    assert G.groebner(FSET, "grlex")["basis"] \
        == G.buchberger(FSET, "grlex")["basis"]
