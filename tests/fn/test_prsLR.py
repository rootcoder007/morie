"""LR(1), SLR(1) and LALR(1) bottom-up parsing."""
import importlib

import pytest

R = importlib.import_module("morie.fn.prsLR")
L = importlib.import_module("morie.fn.prsLL")

G = L.grammar([("E", ["E", "-", "T"]), ("E", ["T"]),
               ("T", ["T", "*", "F"]), ("T", ["F"]),
               ("F", ["(", "E", ")"]), ("F", ["id"])], "E")
W = L.grammar([("S", ["L", "=", "R"]), ("S", ["R"]),
               ("L", ["*", "R"]), ("L", ["id"]),
               ("R", ["L"])], "S")
AMB = L.grammar([("E", ["E", "+", "E"]), ("E", ["id"])], "E")


def _ev(node, vals):
    s, k = node["symbol"], node["children"]
    if s == "F":
        return vals.pop(0) if k[0]["symbol"] == "id" else _ev(k[1],
                                                              vals)
    if len(k) == 1:
        return _ev(k[0], vals)
    a, b = _ev(k[0], vals), _ev(k[2], vals)
    return a - b if k[1]["symbol"] == "-" else a * b


def test_left_recursion_is_lr1_but_not_ll1():
    assert R.conflicts(G, "lr1")["ok"]
    assert not L.is_ll1(G)["ll1"]


def test_subtraction_is_left_associative():
    assert _ev(R.parse(G, ["id", "-", "id", "-", "id"]),
               [1, 2, 3]) == -4


@pytest.mark.parametrize("toks,nums,want", [
    (["id", "-", "id", "*", "id"], [10, 2, 3], 4),
    (["(", "id", "-", "id", ")", "*", "id"], [10, 2, 3], 24),
    (["id", "*", "id", "*", "id"], [2, 3, 4], 24),
    (["id"], [7], 7),
])
def test_the_tree_encodes_precedence(toks, nums, want):
    assert _ev(R.parse(G, toks), list(nums)) == want


def test_the_witness_grammar_is_lr1():
    assert R.conflicts(W, "lr1")["ok"]


def test_but_not_slr1():
    c = R.conflicts(W, "slr1")
    assert not c["ok"]
    assert any(x["kind"] == "shift/reduce" and x["lookahead"] == "="
               for x in c["conflicts"])


def test_canonical_lr1_pays_in_states():
    assert R.conflicts(W, "lr1")["n_states"] \
        > R.conflicts(W, "slr1")["n_states"]


def test_lalr1_is_small_and_strong_here():
    cl, cs = R.conflicts(W, "lalr1"), R.conflicts(W, "slr1")
    assert cl["ok"]
    assert cl["n_states"] == cs["n_states"]


def test_a_conflicted_table_is_not_run():
    with pytest.raises(ValueError):
        R.parse(W, ["id", "=", "id"], "slr1")
    assert L.linearise(R.parse(W, ["id", "=", "id"], "lr1")) \
        == ["id", "=", "id"]


def test_an_ambiguous_grammar_conflicts():
    assert not R.conflicts(AMB, "lr1")["ok"]
    with pytest.raises(ValueError):
        R.parse(AMB, ["id"])


TOKSETS = [["id"], ["id", "-", "id"], ["id", "*", "id"],
           ["(", "id", ")"], ["id", "-", "id", "*", "id"],
           ["(", "id", "-", "id", ")", "*", "(", "id", ")"]]


@pytest.mark.parametrize("toks", TOKSETS)
def test_the_three_methods_agree(toks):
    assert R.parse(G, toks, "lr1") == R.parse(G, toks, "slr1") \
        == R.parse(G, toks, "lalr1")


@pytest.mark.parametrize("toks", TOKSETS)
def test_the_ll_and_lr_parsers_accept_the_same_strings(toks):
    gll = L.grammar([("E", ["T", "E'"]),
                     ("E'", ["-", "T", "E'"]), ("E'", []),
                     ("T", ["F", "T'"]),
                     ("T'", ["*", "F", "T'"]), ("T'", []),
                     ("F", ["(", "E", ")"]), ("F", ["id"])], "E")
    assert L.linearise(L.parse(gll, toks)) \
        == L.linearise(R.parse(G, toks)) == toks


@pytest.mark.parametrize("toks", [["id", "-"], ["-", "id"],
                                  ["id", "id"], ["(", "id"], [],
                                  ["id", ")"]])
def test_malformed_input_is_rejected(toks):
    with pytest.raises(ValueError):
        R.parse(G, toks)


def test_the_augmented_grammar_adds_one_production():
    ag = R.augment(G)
    assert len(ag["rules"]) == len(G["rules"]) + 1
    assert ag["rules"][0] == ("S'", ("E",))


def test_the_start_state_closes_over_every_production():
    ag = R.augment(G)
    col = R.canonical_collection(ag, 1)
    assert len({i for i, _, _ in col["states"][0]}) \
        == len(ag["rules"])


def test_goto_on_an_impossible_symbol_is_empty():
    ag = R.augment(G)
    col = R.canonical_collection(ag, 1)
    assert R.goto(col["states"][0], ")", ag, col["first"],
                  col["nonterminals"], 1) == frozenset()


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError):
        R.build_tables(G, "peg")


def test_the_entry_point_reports_its_method():
    r = R.lr_parser(G, ["id"], "lalr1")
    assert r["method"] == "lalr1"
    assert r["yield"] == ["id"]
