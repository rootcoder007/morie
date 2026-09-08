"""Term rewriting, critical pairs and Knuth-Bendix completion."""
import importlib

import pytest

T = importlib.import_module("morie.fn.trmRew")
U = importlib.import_module("morie.fn.unifAlg")
v, a, c = U.var, U.app, U.const


def m(p, q):
    return a("*", p, q)


def inv(p):
    return a("i", p)


E = c("e")
X, Y, Z = v("x"), v("y"), v("z")
AX = [(m(E, X), X), (m(inv(X), X), E), (m(m(X, Y), Z), m(X, m(Y, Z)))]
PREC = {"i": 3, "*": 2, "e": 1}
G = T.complete(AX, PREC)["rules"]


def test_completion_finds_the_ten_rule_group_system():
    r = T.complete(AX, PREC)
    assert r["complete"]
    assert r["n_rules"] == 10


def test_the_completed_system_is_confluent():
    assert T.is_confluent(G, PREC)["confluent"]


def test_the_raw_axioms_are_not():
    raw = [T.rule(l, r) for l, r in AX]
    assert not T.is_locally_confluent(raw)["locally_confluent"]


@pytest.mark.parametrize("lhs,rhs", [
    (inv(inv(X)), X),
    (inv(E), E),
    (m(X, E), X),
    (m(X, inv(X)), E),
    (inv(m(X, Y)), m(inv(Y), inv(X))),
])
def test_it_decides_group_identities(lhs, rhs):
    assert T.decides(lhs, rhs, G)["equal"]


def test_it_does_not_prove_commutativity():
    assert not T.decides(m(X, Y), m(Y, X), G)["equal"]


def test_the_two_strategies_agree_on_a_convergent_system():
    t = m(m(inv(m(X, Y)), m(X, Y)), inv(E))
    assert T.normal_form(t, G, "innermost")["normal_form"] \
        == T.normal_form(t, G, "outermost")["normal_form"]


def test_non_confluence_is_detected():
    bad = [T.rule(c("a"), c("b")), T.rule(c("a"), c("d"))]
    lc = T.is_locally_confluent(bad)
    assert not lc["locally_confluent"]
    assert lc["n_critical_pairs"] > 0
    assert T.normal_form(c("a"), bad)["normal_form"] \
        != T.normal_form(c("a"), list(reversed(bad)))["normal_form"]


def test_the_lpo_orients_the_completed_rules():
    assert T.is_terminating(G, PREC)["terminating"]


def test_a_growing_rule_is_not_oriented():
    loop = [(a("f", X), a("f", a("f", X)))]
    assert not T.is_terminating(loop, {"f": 1})["terminating"]
    with pytest.raises(ValueError):
        T.normal_form(a("f", c("a")), loop, max_steps=200)


def test_lpo_basics():
    assert T.lpo_greater(m(X, Y), X, PREC)
    assert T.lpo_greater(inv(c("a")), m(c("a"), c("a")), PREC)
    assert not T.lpo_greater(m(c("a"), c("a")), inv(c("a")), PREC)
    assert not T.lpo_greater(m(X, Y), m(X, Y), PREC)


def test_an_unorientable_equation_is_reported():
    f = T.complete([(m(X, Y), m(Y, X))], PREC)
    assert not f["complete"]
    assert "unorientable" in f["reason"]


def test_positions_and_replacement_round_trip():
    t = m(inv(X), m(Y, E))
    assert len(T.positions(t)) == 6
    for p in T.positions(t):
        assert T.replace_at(t, p, T.subterm_at(t, p)) == t
    assert T.replace_at(t, (1, 0), E) == m(inv(X), m(E, E))


def test_a_rewrite_step_reports_where_it_fired():
    st = T.rewrite_step(m(E, c("a")), G)
    assert st["term"] == c("a")
    assert st["position"] == ()
    assert T.rewrite_step(c("a"), G) is None


def test_the_entry_point_carries_the_trace():
    r = T.term_rewriting(m(E, m(E, c("a"))), G)
    assert r["normal_form"] == c("a")
    assert r["steps"] == len(r["trace"]) == 2


@pytest.mark.parametrize("call", [
    lambda: T.rule(v("x"), c("e")),
    lambda: T.rule(m(v("x"), c("e")), m(v("x"), v("y"))),
    lambda: T.subterm_at(m(v("x"), c("e")), (9,)),
    lambda: T.replace_at(m(v("x"), c("e")), (9,), c("e")),
    lambda: T.normal_form(c("a"), G, "clever"),
])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call()
