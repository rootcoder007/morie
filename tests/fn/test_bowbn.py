"""Tests for bowbn.bowarc (Brito and Pearl bow-free rule)."""

from morie.fn.bowbn import bow_ban_theorem, bowarc

# Z confounds X and Y, and X -> Y directly
DAG = {"Z": ["X", "Y"], "X": ["Y"], "Y": []}


def test_bowbn_basic():
    """A direct edge plus correlated errors on the same pair is a bow."""
    assert bow_ban_theorem is bowarc
    res = bowarc(DAG, [("X", "Y")], "X", "Y")
    assert res["direct"] is True
    assert res["confounded"] is True
    assert res["isbow"] is True
    assert res["bows"] == [("X", "Y")]
    assert res["nbows"] == 1
    assert res["bowfree"] is False
    assert res["acyclic"] is True
    # Theorem 4: bow-free AND acyclic, so a bow blocks identification
    assert res["identified"] is False
    assert res["method"] == "Bow (confounded parent-child pair) test"


def test_bowbn_edge():
    """No bow-arc leaves the acyclic model identified; a cycle does not."""
    free = bowarc(DAG, [], "X", "Y")
    assert free["direct"] is True
    assert free["confounded"] is False
    assert free["isbow"] is False
    assert free["nbows"] == 0
    assert free["bows"] == []
    assert free["bowfree"] is True
    assert free["acyclic"] is True
    assert free["identified"] is True
    # correlated errors between a non-adjacent pair are not a bow
    far = bowarc({"X": ["Y"], "Y": [], "Z": []}, [("X", "Z")], "X", "Y")
    assert far["direct"] is True
    assert far["confounded"] is False
    assert far["isbow"] is False
    assert far["bowfree"] is True
    assert far["identified"] is True
    # a pair with correlated errors but no edge between them
    rev = bowarc(DAG, [("X", "Y")], "Y", "X")
    assert rev["direct"] is False
    assert rev["confounded"] is True
    assert rev["isbow"] is False
    # the bow is still counted for the model as a whole
    assert rev["nbows"] == 1
    assert rev["identified"] is False
    # Theorem 4 needs acyclicity too
    cyc = bowarc({"A": ["B"], "B": ["A"]}, [], "A", "B")
    assert cyc["acyclic"] is False
    assert cyc["bowfree"] is True
    assert cyc["identified"] is False
