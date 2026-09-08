"""Tests for agpar."""

from morie.fn import _array_core as np
import pytest

from morie.fn.agpar import party_unity_score


def test_agpar_basic():
    V = np.array([[1, 1], [1, 1], [1, 0], [0, 0], [0, 0]], dtype=float)
    pid = np.array(["a", "a", "a", "b", "b"])
    out = party_unity_score(V, pid)
    assert out["unity"][2] == pytest.approx(0.5)
    assert out["by_party"]["b"] == pytest.approx(1.0)


def test_agpar_edge():
    V = np.array([[1, 1], [0, 0]], dtype=float)
    with pytest.raises(ValueError):
        party_unity_score(V, ["a", "a", "b"])  # length mismatch
    with pytest.raises(ValueError):
        party_unity_score(V, ["a", "a"], unity_votes_only=True)  # one party
