"""Tests for ladrc.ladder_of_causation."""

import pytest

from morie.fn.ladrc import ladder_of_causation


def test_ladrc_basic():
    """Table 2.1: association needs only the joint distribution, intervention
    a causal graph, counterfactuals a full SCM."""
    got = [(ladder_of_causation(k)["needsgraph"], ladder_of_causation(k)["needsscm"]) for k in (1, 2, 3)]
    assert got == [(False, False), (True, False), (True, True)]
    assert [ladder_of_causation(k)["level"] for k in (1, 2, 3)] == [1, 2, 3]


def test_ladrc_edge():
    with pytest.raises(ValueError, match="rung"):
        ladder_of_causation(4)


