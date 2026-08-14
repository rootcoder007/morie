"""LL(1) top-down parsing."""
import importlib

import pytest

L = importlib.import_module("morie.fn.prsLL")

LREC = L.grammar([("E", ["E", "+", "T"]), ("E", ["T"]),
                  ("T", ["T", "*", "F"]), ("T", ["F"]),
                  ("F", ["(", "E", ")"]), ("F", ["id"])], "E")
G = L.grammar([("E", ["T", "E'"]),
               ("E'", ["+", "T", "E'"]), ("E'", []),
               ("T", ["F", "T'"]),
               ("T'", ["*", "F", "T'"]), ("T'", []),
               ("F", ["(", "E", ")"]), ("F", ["id"])], "E")


def test_first_sets():
    f = L.first_sets(G)
    assert f["E"] == f["T"] == f["F"] == {"(", "id"}
    assert f["E'"] == {"+", ""}
    assert f["T'"] == {"*", ""}


def test_follow_sets():
    fo = L.follow_sets(G)
    assert fo["E"] == fo["E'"] == {")", "$"}
    assert fo["T"] == fo["T'"] == {"+", ")", "$"}
    assert fo["F"] == {"*", "+", ")", "$"}


def test_the_right_recursive_grammar_is_ll1():
    assert L.is_ll1(G)["ll1"]


def test_the_left_recursive_one_is_not():
    r = L.is_ll1(LREC)
    assert not r["ll1"]
    assert len(r["conflicts"]) >= 2


def test_left_recursion_is_detected():
    assert L.left_recursive(LREC) == ["E", "T"]
    assert L.left_recursive(G) == []


def test_indirect_left_recursion_is_detected():
    ind = L.grammar([("A", ["B", "x"]), ("B", ["A", "y"]),
                     ("A", ["z"]), ("B", ["w"])], "A")
    assert set(L.left_recursive(ind)) == {"A", "B"}


def test_removing_left_recursion_gives_an_ll1_grammar():
    fixed = L.remove_left_recursion(LREC)
    assert L.is_ll1(fixed)["ll1"]
    assert L.left_recursive(fixed) == []


def test_the_two_routes_build_the_same_tree():
    toks = ["id", "+", "id", "*", "id"]
    assert L.parse(G, toks, "table") \
        == L.parse(G, toks, "recursive_descent")


def test_the_tree_yields_its_input():
    toks = ["id", "+", "(", "id", "*", "id", ")"]
    assert L.linearise(L.parse(G, toks)) == toks


def _ev(node, vals):
    s, k = node["symbol"], node["children"]
    if s == "F":
        return vals.pop(0) if k[0]["symbol"] == "id" else _ev(k[1],
                                                              vals)
    if s in ("T", "E"):
        acc = _ev(k[0], vals)
        tail, op = k[1], ("*" if s == "T" else "+")
        while tail["children"]:
            rhs = _ev(tail["children"][1], vals)
            acc = acc * rhs if op == "*" else acc + rhs
            tail = tail["children"][2]
        return acc
    raise AssertionError(s)


@pytest.mark.parametrize("toks,nums,want", [
    (["id", "+", "id", "*", "id"], [1, 2, 3], 7),
    (["(", "id", "+", "id", ")", "*", "id"], [1, 2, 3], 9),
    (["id", "*", "id", "+", "id"], [2, 3, 4], 10),
    (["id", "+", "id", "+", "id"], [1, 2, 3], 6),
    (["id"], [42], 42),
])
def test_the_tree_encodes_precedence(toks, nums, want):
    assert _ev(L.parse(G, toks), list(nums)) == want


@pytest.mark.parametrize("toks", [
    ["id", "+"], ["+", "id"], ["id", "id"], ["(", "id"], [],
    ["id", ")"],
])
def test_malformed_input_is_rejected(toks):
    with pytest.raises(ValueError):
        L.parse(G, toks)


def test_a_non_ll1_grammar_is_not_parsed():
    with pytest.raises(ValueError):
        L.parse(LREC, ["id"])


@pytest.mark.parametrize("call", [
    lambda: L.grammar([]),
    lambda: L.grammar([("E", ["id"])], "S"),
    lambda: L.grammar([("E", ["id"]), ("X", ["y"])], "E"),
    lambda: L.grammar([("E", ["$"])]),
    lambda: L.parse(G, ["id"], "psychic"),
    lambda: L.remove_left_recursion(L.grammar([("A", ["A", "x"])],
                                              "A")),
])
def test_bad_grammars_are_refused(call):
    with pytest.raises(ValueError):
        call()


def test_terminals_and_nonterminals_are_separated():
    assert set(L.nonterminals(G)) == {"E", "E'", "T", "T'", "F"}
    assert set(L.terminals(G)) == {"+", "*", "(", ")", "id"}


def test_the_entry_point_reports_its_route():
    r = L.ll_parser(G, ["id"], "recursive_descent")
    assert r["route"] == "recursive_descent"
    assert r["yield"] == ["id"]
