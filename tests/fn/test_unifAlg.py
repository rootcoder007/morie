"""Robinson unification."""
import importlib

import pytest

U = importlib.import_module("morie.fn.unifAlg")
v, a, c = U.var, U.app, U.const

T1 = a("f", v("x"), a("g", v("y")))
T2 = a("f", a("h", v("z")), a("g", c("a")))
OVER = ["x", "y", "z"]


def test_textbook_pair_gives_the_expected_mgu():
    r = U.unify(T1, T2)
    assert r["unified"]
    assert r["mgu"] == {"x": a("h", v("z")), "y": c("a")}


def test_the_mgu_actually_unifies():
    m = U.unify(T1, T2)["mgu"]
    assert U.apply_subst(T1, m) == U.apply_subst(T2, m)


def test_every_other_unifier_factors_through_the_mgu():
    theta = {"x": a("h", c("b")), "y": c("a"), "z": c("b")}
    assert U.apply_subst(T1, theta) == U.apply_subst(T2, theta)
    m = U.unify(T1, T2)["mgu"]
    d = U.factor_through(m, theta, OVER)
    assert d is not None
    for n in OVER:
        assert (U.apply_subst(U.apply_subst(v(n), m), d)
                == U.apply_subst(v(n), theta))


def test_the_mgu_is_idempotent():
    m = U.unify(T1, T2)["mgu"]
    for n in OVER:
        assert U.apply_subst(U.apply_subst(v(n), m), m) \
            == U.apply_subst(v(n), m)


@pytest.mark.parametrize("t", [a("f", v("x")), a("f", a("g", v("x")))])
def test_occurs_check_refuses_a_variable_inside_its_own_binding(t):
    r = U.unify(v("x"), t)
    assert not r["unified"]
    assert "occurs check" in r["reason"]


def test_suppressing_the_occurs_check_reports_the_cycle():
    r = U.unify(v("x"), a("f", v("x")), occurs_check=False)
    assert r["unified"] and r["cyclic"]
    assert r["mgu"]["x"] == a("f", v("x"))


@pytest.mark.parametrize("p,q", [
    (a("f", v("x")), a("g", v("x"))),
    (a("f", v("x")), a("f", v("x"), v("y"))),
    (c("a"), c("b")),
    (a("f", v("x"), v("x")), a("f", c("a"), c("b"))),
])
def test_terms_that_cannot_unify(p, q):
    assert not U.unify(p, q)["unified"]


def test_a_repeated_variable_propagates():
    r = U.unify(a("f", v("x"), v("x")), a("f", v("y"), c("a")))
    assert r["unified"]
    assert U.apply_subst(v("x"), r["mgu"]) == c("a")
    assert U.apply_subst(v("y"), r["mgu"]) == c("a")


def test_unification_is_symmetric():
    s1, s2 = U.unify(T1, T2)["mgu"], U.unify(T2, T1)["mgu"]
    for n in OVER:
        assert U.apply_subst(v(n), s1) == U.apply_subst(v(n), s2)


def test_identical_terms_need_no_bindings():
    assert U.unify(T1, T1)["mgu"] == {}


def test_a_bare_variable_unifies_with_anything():
    assert U.unify(v("q"), T2)["mgu"] == {"q": T2}


def test_matching_is_one_way():
    assert U.match(a("f", v("x")), a("f", c("a"))) == {"x": c("a")}
    assert U.match(a("f", c("a")), a("f", v("x"))) is None


def test_matching_respects_a_repeated_pattern_variable():
    assert U.match(a("f", v("x"), v("x")), a("f", c("a"), c("b"))) \
        is None
    assert U.match(a("f", v("x"), v("x")), a("f", c("a"), c("a"))) \
        == {"x": c("a")}


def test_composition_applies_the_inner_substitution_first():
    s = U.compose({"y": c("a")}, {"x": a("g", v("y"))})
    assert U.apply_subst(v("x"), s) == a("g", c("a"))


def test_variables_are_reported_in_first_seen_order():
    assert U.variables(a("f", v("b"), a("g", v("a"), v("b")))) \
        == ["b", "a"]


def test_disagreement_finds_the_leftmost_difference():
    d = U.disagreement(a("f", c("a"), v("x")), a("f", c("a"), c("b")))
    assert d == (v("x"), c("b"))
    assert U.disagreement(T1, T1) is None


@pytest.mark.parametrize("bad", [("VAR",), ("APP", "f"), "f(x)"])
def test_malformed_terms_are_refused(bad):
    with pytest.raises(ValueError):
        U.variables(bad)


def test_the_alias_resolves_to_unify():
    assert U.unification is U.unify
