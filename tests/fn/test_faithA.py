"""Tests for faithA: the faithfulness assumption for one triple.

Faithfulness says X indep_P Y | Z implies X indep_G Y | Z (a
distributional independence must be a d-separation); the global Markov
property is the converse. Checked on the three canonical graphs, whose
d-separations are known in closed form. Edges are (parent, child).
"""

from morie.fn.faithA import faithchk, faithfulness_assumption

CHAIN = [("A", "B"), ("B", "C")]        # A -> B -> C
FORK = [("B", "A"), ("B", "C")]         # A <- B -> C
COLLIDER = [("A", "B"), ("C", "B")]     # A -> B <- C


def _sep(g, z):
    return faithchk(g, "A", "C", z)["dseparated"]


def test_faithA_basic():
    """Chain and fork: A, C dependent marginally, separated given B."""
    for g in (CHAIN, FORK):
        assert _sep(g, ()) is False
        assert _sep(g, ("B",)) is True


def test_a_collider_reverses_the_pattern():
    """Conditioning on a collider OPENS the path."""
    assert _sep(COLLIDER, ()) is True
    assert _sep(COLLIDER, ("B",)) is False
    # conditioning on a descendant of the collider also opens it
    assert _sep(COLLIDER + [("B", "D")], ("D",)) is False


def test_faithfulness_and_markov_follow_their_definitions():
    for g in (CHAIN, FORK, COLLIDER):
        for z in ((), ("B",)):
            sep = _sep(g, z)
            for indep in (True, False):
                r = faithchk(g, "A", "C", z, indep=indep)
                # faithful: indep_P => indep_G ; markov: indep_G => indep_P
                assert r["faithful"] is ((not indep) or sep)
                assert r["markov"] is ((not sep) or indep)
    # the textbook violation: an observed independence the graph lacks
    v = faithchk(CHAIN, "A", "C", (), indep=True)
    assert v["faithful"] is False and v["markov"] is True
    assert faithfulness_assumption is faithchk


def test_faithA_edge():
    """A dict of children is the same graph as the edge list."""
    as_dict = {"A": ["B"], "B": ["C"], "C": []}
    for z in ((), ("B",)):
        assert faithchk(as_dict, "A", "C", z)["dseparated"] == _sep(CHAIN, z)
