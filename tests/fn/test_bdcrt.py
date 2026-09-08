"""Tests for bdcrt.backdoor_criterion (Pearl 2009, Def. 3.3.1)."""

import pytest

from morie.fn.bdcrt import backdoor_criterion

# Classic confounder: Z -> X, Z -> Y, X -> Y.
CONFOUND = {"Z": ["X", "Y"], "X": ["Y"]}
# Collider: X -> C <- Y, plus X -> Y.
COLLIDER = {"X": ["C", "Y"], "Y": ["C"]}
# Mediator: X -> M -> Y.
MEDIATOR = {"X": ["M"], "M": ["Y"]}


def test_confounder_must_be_adjusted_for():
    assert not backdoor_criterion(CONFOUND, "X", "Y", [])["satisfied"]
    assert backdoor_criterion(CONFOUND, "X", "Y", ["Z"])["satisfied"]


def test_the_open_backdoor_path_is_named():
    res = backdoor_criterion(CONFOUND, "X", "Y", [])
    assert res["n_backdoor"] == 1
    assert len(res["open_paths"]) == 1
    assert "remain open" in res["reason"]


def test_adjusting_for_a_mediator_is_rejected_as_a_descendant():
    """M is a descendant of X, so conditioning on it removes part of the
    effect being estimated -- the criterion forbids it outright."""
    res = backdoor_criterion(MEDIATOR, "X", "Y", ["M"])
    assert not res["satisfied"]
    assert res["descendant_violations"] == ["M"]
    assert "descendants of X" in res["reason"]


def test_empty_set_is_valid_when_there_is_no_confounding():
    assert backdoor_criterion(MEDIATOR, "X", "Y", [])["satisfied"]


def test_conditioning_on_a_collider_opens_a_path():
    """The rule that makes d-separation more than 'adjust for everything'.

    X -> C <- Y has no open back-door path on its own, but C is a
    descendant of X, so the criterion refuses it.
    """
    bare = backdoor_criterion(COLLIDER, "X", "Y", [])
    assert bare["satisfied"]
    withc = backdoor_criterion(COLLIDER, "X", "Y", ["C"])
    assert not withc["satisfied"]


def test_collider_descendant_also_opens_the_path():
    """U -> X, U -> C, W -> C, W -> Y, plus C -> D. Conditioning on D,
    a descendant of the collider C, unblocks X <- U -> C <- W -> Y."""
    g = {"U": ["X", "C"], "W": ["C", "Y"], "C": ["D"], "X": ["Y"]}
    assert not backdoor_criterion(g, "X", "Y", ["D"])["satisfied"]
    assert backdoor_criterion(g, "X", "Y", ["U"])["satisfied"]


def test_a_chain_confounder_can_be_blocked_at_either_link():
    """X <- A <- B -> Y: blocking at A or at B both suffice."""
    g = {"B": ["A", "Y"], "A": ["X"], "X": ["Y"]}
    assert not backdoor_criterion(g, "X", "Y", [])["satisfied"]
    assert backdoor_criterion(g, "X", "Y", ["A"])["satisfied"]
    assert backdoor_criterion(g, "X", "Y", ["B"])["satisfied"]


def test_edge_list_input_is_accepted():
    edges = [("Z", "X"), ("Z", "Y"), ("X", "Y")]
    assert backdoor_criterion(edges, "X", "Y", ["Z"])["satisfied"]


def test_validates_inputs():
    with pytest.raises(ValueError, match="not a node"):
        backdoor_criterion(CONFOUND, "Q", "Y", [])
    with pytest.raises(ValueError, match="nodes not in the graph"):
        backdoor_criterion(CONFOUND, "X", "Y", ["Q"])
    with pytest.raises(ValueError, match="must not contain X or Y"):
        backdoor_criterion(CONFOUND, "X", "Y", ["X"])
    with pytest.raises(ValueError, match="cycle"):
        backdoor_criterion({"X": ["Y"], "Y": ["Z"], "Z": ["X"]}, "X", "Y", [])
