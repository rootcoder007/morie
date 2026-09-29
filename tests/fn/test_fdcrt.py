"""Tests for morie.fn.fdcrt: Pearl's three conditions on textbook graphs."""

from morie.fn.fdcrt import frontdoor_criterion


def test_smoking_tar_cancer():
    g = {"U": ["X", "Y"], "X": ["Z"], "Z": ["Y"]}
    r = frontdoor_criterion(g, "X", "Y", "Z")
    assert r["satisfied"] and r["cond1"] and r["cond2"] and r["cond3"]


def test_violations():
    # a direct X -> Y edge bypasses Z
    assert not frontdoor_criterion({"X": ["Z", "Y"], "Z": ["Y"]}, "X", "Y", "Z")["cond1"]
    # a confounder of X and Z opens a back-door from X to Z
    r = frontdoor_criterion({"U": ["X", "Z"], "X": ["Z"], "Z": ["Y"]}, "X", "Y", "Z")
    assert r["cond1"] and not r["cond2"]
    # a confounder of Z and Y not blocked by X
    r = frontdoor_criterion({"V": ["Z", "Y"], "X": ["Z"], "Z": ["Y"]}, "X", "Y", "Z")
    assert r["cond1"] and r["cond2"] and not r["cond3"]
