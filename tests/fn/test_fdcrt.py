"""Tests for fdcrt.frontdoor_criterion."""

import pytest

from morie.fn.fdcrt import frontdoor_criterion

# X <- U -> Y, X -> Z -> Y: Z is a valid front-door mediator
GOOD = {"U": ["X", "Y"], "X": ["Z"], "Z": ["Y"]}


def test_fdcrt_basic():
    out = frontdoor_criterion(GOOD, "X", "Y", ("Z",))
    assert out["satisfied"] is True
    assert out["cond1"] and out["cond2"] and out["cond3"]


def test_fdcrt_edge():
    # a direct X -> Y edge leaves a path Z does not intercept
    leaky = {"U": ["X", "Y"], "X": ["Z", "Y"], "Z": ["Y"]}
    assert frontdoor_criterion(leaky, "X", "Y", ("Z",))["satisfied"] is False
    with pytest.raises(ValueError):
        frontdoor_criterion(GOOD, "X", "Y", ("W",))  # W not in the graph
