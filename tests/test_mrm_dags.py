# SPDX-License-Identifier: AGPL-3.0-or-later
"""Causal DAGs for MRM, and the bundled structures.

Parity with rmorie's R/dag_native.R. The back-door assertions are
graph-theory facts about these specific graphs, independently checkable
by hand, so they can fail.
"""

import pytest

import morie


def test_edges_parse_from_arrows_or_pairs():
    a = morie.causal_dag(["x -> y"], exposure="x", outcome="y")
    b = morie.causal_dag([("x", "y")], exposure="x", outcome="y")
    assert a.edges == b.edges == [("x", "y")]
    assert a.nodes == ["x", "y"]
    # whitespace around the arrow is not significant
    c = morie.causal_dag(["  x->y  "], exposure="x", outcome="y")
    assert c.edges == [("x", "y")]
    # a lone string is one edge, not a sequence of characters
    d = morie.causal_dag("x -> y", exposure="x", outcome="y")
    assert d.edges == [("x", "y")]


def test_a_cycle_is_rejected():
    with pytest.raises(ValueError, match="cycle"):
        morie.causal_dag(["a -> b", "b -> c", "c -> a"],
                         exposure="a", outcome="c")
    # a diamond is acyclic and must be accepted
    g = morie.causal_dag(["a -> b", "a -> c", "b -> d", "c -> d"],
                         exposure="a", outcome="d")
    assert len(g.nodes) == 4


def test_exposure_and_outcome_must_be_in_the_graph():
    with pytest.raises(ValueError, match="exposure not in graph"):
        morie.causal_dag(["a -> b"], exposure="zzz", outcome="b")
    with pytest.raises(ValueError, match="outcome not in graph"):
        morie.causal_dag(["a -> b"], exposure="a", outcome="zzz")
    with pytest.raises(ValueError, match="at least one edge"):
        morie.causal_dag([], exposure="a", outcome="b")
    with pytest.raises(ValueError, match="'A -> B'"):
        morie.causal_dag(["a ~ b"], exposure="a", outcome="b")


def test_parents_and_children():
    g = morie.mrm_dags()["placement"]
    assert sorted(g.parents("outcome")) == \
        ["age", "placement", "prior_record", "race"]
    assert g.children("race") == ["placement", "outcome"]
    assert g.parents("race") == []


def test_bundled_dags_match_rmorie():
    d = morie.mrm_dags()
    assert sorted(d) == ["placement", "use_of_force"]

    p = d["placement"]
    assert p.exposure == "placement" and p.outcome == "outcome"
    assert sorted(p.nodes) == ["age", "outcome", "placement",
                               "prior_record", "race"]
    assert sorted("%s>%s" % e for e in p.edges) == [
        "age>outcome", "age>placement", "placement>outcome",
        "prior_record>outcome", "prior_record>placement",
        "race>outcome", "race>placement"]

    u = d["use_of_force"]
    assert u.exposure == "police_contact" and u.outcome == "force"
    assert sorted(u.nodes) == ["force", "neighbourhood", "police_contact",
                               "race"]
    assert sorted("%s>%s" % e for e in u.edges) == [
        "neighbourhood>force", "neighbourhood>police_contact",
        "police_contact>force", "race>force", "race>police_contact"]


def test_the_common_causes_are_what_must_be_adjusted():
    # THE SUBSTANCE of both bundled graphs: the confounders are common
    # causes of exposure and outcome, so the unadjusted comparison is
    # confounded and the full set closes the back doors.
    p = morie.mrm_dags()["placement"]
    assert p.backdoor(()).satisfied is False
    assert p.backdoor(("race", "prior_record", "age")).satisfied is True
    # a proper subset leaves a back door open
    assert p.backdoor(("race",)).satisfied is False

    u = morie.mrm_dags()["use_of_force"]
    assert u.backdoor(()).satisfied is False
    assert u.backdoor(("neighbourhood", "race")).satisfied is True
    # a bare string is one node, not a sequence of characters
    assert u.backdoor("race").satisfied is False


def test_edges_are_the_representation_the_rest_of_morie_takes():
    # dag_plot and fn.bdcrt both take [(from, to)], so a graph built here
    # needs no conversion.
    g = morie.mrm_dags()["placement"]
    assert all(isinstance(e, tuple) and len(e) == 2 for e in g.edges)
    from morie.fn.bdcrt import backdoor_criterion
    assert backdoor_criterion(g.edges, g.exposure, g.outcome,
                              ("race", "prior_record", "age")).satisfied
